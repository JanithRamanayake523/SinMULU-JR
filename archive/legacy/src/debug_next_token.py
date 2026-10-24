"""Diagnostic helpers: what does the model actually want to output right after the prompt, and how is the text tokenised?
DEV DATA ONLY (labelled public train). Never run this on the public test set.

Model / options
  preset(key)                              -> model id, dtype, chat_capable from configs/<key>.yaml
  load(model_id, dtype)                    -> tok, model
Tokenisation (no model needed except load)
  show_prompt_tokens(tok, row, ...)        -> how the exact prompt is tokenised (parts, head/tail tokens, special tokens)
  word_view(tok, text, n_words)            -> per word: how many tokens, which pieces (fertility on a sample)
  answer_tokenization(tok)                 -> how '1', ' 1', 'පිළිතුර:' ... are tokenised (what the scorer has to hit)
  compare_tokenizers(model_ids, text)      -> the same text through several tokenizers (tokenizers only, no GPU)
Scoring and next-token behaviour
  scoring_table(tok, model, rows, ...)     -> the REAL scoring path with chosen prompt / chat / permutations / calibration
  show_question(tok, model, row, ...)      -> exact text, top-k next tokens, digit candidates, calibration, greedy continuation
  summary(tok, model, rows, ...)           -> most common top-1 next tokens and mean digit mass
"""
import collections, math, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import eval_logprob as E
from data_utils import ROOT, cyclic_perms, permute_options

DIGITS = E.DIGITS
PRESET_KEYS = {"qwen": "qwen3_0.6b", "gemma": "gemma3_1b", "llama": "llama32_1b",
               "qwen4b": "qwen3_4b_instruct_2507", "gemma4b": "gemma3_4b", "llama3b": "llama32_3b"}


# ----------------------------------------------------------------------------- model / options
def preset(key):
    """key: 'qwen' | 'gemma' | 'llama' (or a config name). Returns dict(model, dtype, chat_capable) from configs/*.yaml."""
    import yaml
    name = PRESET_KEYS.get(key, key)
    cfg = yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8"))
    return {"model": cfg["model"], "dtype": cfg.get("dtype", "fp16"), "chat_capable": bool(cfg.get("chat_capable"))}


def load(model_id, dtype="fp16", revision=None):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    kw = dict(revision=revision, device_map="auto" if torch.cuda.is_available() else None)
    try:
        model = E.load_causal_lm(model_id, dtype=dt, **kw)  # transformers >= 5
    except TypeError:
        model = E.load_causal_lm(model_id, torch_dtype=dt, **kw)
    model.eval()
    return tok, model


# ----------------------------------------------------------------------------- shared helpers
def _ids(tok, text, chat):
    return tok(text, add_special_tokens=not chat, return_tensors="pt")["input_ids"]


def _next_logprobs(model, ids):
    import torch
    with torch.no_grad():
        logits = model(input_ids=ids.to(next(model.parameters()).device)).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1).cpu()


def _digit_ids(tok):
    return {tok.encode(d, add_special_tokens=False)[0] for d in DIGITS if len(tok.encode(d, add_special_tokens=False)) == 1}


def _ordered_row(row, perm_index):
    """Row with its options in the chosen cyclic order (answer remapped). perm_index 0 = original order."""
    perm = cyclic_perms(4)[perm_index]
    return permute_options(row, perm), perm


def build_text(tok, row, prompt_mode="nosubject", chat=False, enable_thinking=False):
    return E.render_prompt(tok, E.prompt_for(row, prompt_mode), chat, enable_thinking)


def _check_chat(tok, chat, capable=None):
    if chat and getattr(tok, "chat_template", None) is None:
        raise ValueError("this model has no chat template: use CHAT = False")
    if chat and capable is False:
        raise ValueError("config says this model is not chat capable: use CHAT = False")


# ----------------------------------------------------------------------------- tokenisation
def _piece(tok, i):
    """(raw vocab piece, decoded text, flag). Byte-level BPE (Qwen, Llama) splits Sinhala characters into bytes, so a
    single token may decode to a replacement character: flagged as 'partial bytes'."""
    raw = tok.convert_ids_to_tokens(int(i))
    dec = tok.decode([int(i)])
    flag = ""
    if "�" in dec:
        flag = "partial bytes"
    elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", str(raw)):
        flag = "byte fallback"
    elif i in getattr(tok, "all_special_ids", []):
        flag = "special"
    return raw, dec, flag


def show_tokenization(tok, text, add_special_tokens=False, head=40, tail=40, title="text"):
    """Token table of `text`: index, id, raw vocab piece, decoded text, flag. Long texts show head and tail only."""
    ids = tok(text, add_special_tokens=add_special_tokens)["input_ids"]
    print(f"--- {title}: {len(ids)} tokens, {len(text)} characters ({len(ids) / max(len(text), 1):.2f} tokens per character)")
    rows = list(enumerate(ids))
    if len(rows) > head + tail:
        shown = rows[:head] + [None] + rows[-tail:]
    else:
        shown = rows
    print(f"{'idx':>5} {'id':>8}  {'raw piece':<18} {'decoded':<14} flag")
    for item in shown:
        if item is None:
            print("   ... (middle omitted) ...")
            continue
        k, i = item
        raw, dec, flag = _piece(tok, i)
        print(f"{k:>5} {i:>8}  {str(raw)!r:<18} {dec!r:<14} {flag}")
    n_partial = sum(1 for i in ids if _piece(tok, i)[2] in ("partial bytes", "byte fallback"))
    print(f"    tokens that are partial bytes / byte fallback: {n_partial} of {len(ids)}")
    return ids


def show_prompt_tokens(tok, row, prompt_mode="nosubject", chat=False, enable_thinking=False, perm_index=0, head=30, tail=40):
    """How the exact prompt is tokenised: tokens per part of the prompt (instruction, question, each option, answer cue),
    then the head and tail tokens of the full text that the model receives."""
    row, perm = _ordered_row(row, perm_index)
    raw_prompt = E.prompt_for(row, prompt_mode)
    text = E.render_prompt(tok, raw_prompt, chat, enable_thinking)
    print(f"=== prompt={prompt_mode} chat={'on' if chat else 'off'} option order={perm} (new position i holds original option perm[i]+1)")
    print("--- tokens per line of the prompt (each line tokenised on its own, so the sum is approximate):")
    total = 0
    for line in raw_prompt.split(chr(10)):
        if not line.strip():
            continue
        n = len(tok(line, add_special_tokens=False)["input_ids"])
        total += n
        print(f"   {n:4d}  {line[:70]!r}")
    print(f"   {total:4d}  (sum over lines)")
    ids = show_tokenization(tok, text, add_special_tokens=not chat, head=head, tail=tail, title="full text fed to the model")
    specials = [(k, tok.decode([i])) for k, i in enumerate(ids) if i in getattr(tok, "all_special_ids", [])]
    print(f"--- special tokens in the input (index, text): {specials}")
    return ids


def word_view(tok, text, n_words=15):
    """Per word: number of tokens and the pieces, using the offsets of the tokenisation of the whole text (so a leading
    space belongs to the word it precedes). Also prints fertility (tokens per word) for the whole text."""
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
    except Exception:
        ids = offs = None
    print(f"--- {len(words)} words; fertility = {len(ids) / max(len(words), 1):.2f} tokens per word" if ids else f"--- {len(words)} words")
    for s, e, w in words[:n_words]:
        if ids is not None:
            mine = [i for i, (a, b) in zip(ids, offs) if b > s and a < e]
        else:
            mine = tok(w, add_special_tokens=False)["input_ids"]
        pieces = " | ".join(tok.decode([i]) for i in mine)
        print(f"   {len(mine):2d} tokens  {w!r:<28} -> {pieces}")


def answer_tokenization(tok):
    """How the answer strings the scorer looks for are tokenised. The scorer needs a SINGLE token for the digit."""
    print(f"{'string':<14} {'ids':<22} pieces")
    for s in ["1", " 1", "2", " 2", "3", " 3", "4", " 4", "10", "පිළිතුර:", "පිළිතුර: 1", "පිළිතුර:" + chr(10) + "1"]:
        ids = tok.encode(s, add_special_tokens=False)
        print(f"{s!r:<14} {str(ids):<22} {[tok.decode([i]) for i in ids]}")
    single = {d: len(tok.encode(d, add_special_tokens=False)) == 1 for d in DIGITS}
    spaced = {d: len(tok.encode(' ' + d, add_special_tokens=False)) == 1 for d in DIGITS}
    print("single-token '1'..'4':", single, "| single-token ' 1'..' 4':", spaced)
    if not all(spaced.values()):
        print("-> ' digit' is TWO tokens (space + digit): the model may emit the space first; the scorer reads the digit at the position BEFORE the space.")


def compare_tokenizers(model_ids, text, n_words=10):
    """Same text through several tokenizers: no model weights, no GPU (gated models need HF_TOKEN)."""
    from transformers import AutoTokenizer
    for mid in model_ids:
        tok = AutoTokenizer.from_pretrained(mid)
        print("=" * 90)
        print(mid, f"| vocab size {len(tok)}")
        word_view(tok, text, n_words)


# ----------------------------------------------------------------------------- next-token behaviour
def digit_candidates(tok, model, ids):
    """Full-sequence log-prob of each candidate answer string ('1' and ' 1'), even if it is several tokens.
    Returns {candidate: (token_ids, logprob)}."""
    import torch
    out = {}
    base = ids
    for d in DIGITS:
        for cand in (d, " " + d):
            c = tok.encode(cand, add_special_tokens=False)
            full = torch.cat([base, torch.tensor([c])], dim=1)
            with torch.no_grad():
                logits = model(input_ids=full.to(next(model.parameters()).device)).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1).cpu()
            n = base.shape[1]
            total = sum(lp[n - 1 + k, c[k]].item() for k in range(len(c)))
            out[cand] = (c, total)
    return out


def _digit_logprobs(tok, model, row, prompt_mode, chat, enable_thinking):
    """Log-probs of the four digits exactly as the scorer computes them (logsumexp over the single-token variants)."""
    import torch
    ids = _ids(tok, build_text(tok, row, prompt_mode, chat, enable_thinking), chat)
    lp = _next_logprobs(model, ids)
    return [torch.logsumexp(lp[v], dim=0).item() for v in E.digit_variants(tok)], lp, ids


def _prior(tok, model, row, prompt_mode, chat, enable_thinking):
    """Content-free prior used for calibration: same inputs and averaging as eval_logprob.score_rows."""
    subj = "" if prompt_mode in ("nosubject", "nosubject_strict") else row.get("subject_si", "N/A")
    per_input = []
    for t in E.CONTENT_FREE:
        dummy = {"question": t, "choices": [t or "-"] * 4, "subject": subj, "subject_si": subj}
        lps, _, _ = _digit_logprobs(tok, model, dummy, prompt_mode, chat, enable_thinking)
        per_input.append(lps)
    return [sum(l[j] for l in per_input) / len(per_input) for j in range(4)], per_input


def show_question(tok, model, row, prompt_mode="nosubject", chat=False, enable_thinking=False, top_k=15, generate=100,
                  perm_index=0, calibrate=False):
    """Everything about the position the scorer reads, for one question in one option order (perm_index 0-3)."""
    import torch
    row, perm = _ordered_row(row, perm_index)
    text = build_text(tok, row, prompt_mode, chat, enable_thinking)
    ids = _ids(tok, text, chat)
    print(f"=== chat={'on' if chat else 'off'}  prompt={prompt_mode}  option order={perm} (index {perm_index})  "
          f"gold answer in this order={row.get('answer')}  ({ids.shape[1]} tokens)")
    print("--- exact text fed to the model (repr, so spaces/newlines/special tokens are visible):")
    print(repr(text[-350:]))
    last = ids[0, -12:].tolist()
    print("--- last 12 input tokens:", [(i, tok.decode([i])) for i in last])
    lp = _next_logprobs(model, ids)
    top = torch.topk(lp, top_k)
    print(f"--- top-{top_k} next tokens:")
    digit_ids = _digit_ids(tok)
    for p, i in zip(top.values.exp().tolist(), top.indices.tolist()):
        mark = "  <-- digit" if i in digit_ids else ""
        print(f"   {p:7.4f}  id={i:<7} {tok.decode([i])!r}{mark}")
    direct = sum(lp[i].exp().item() for i in digit_ids)
    print(f"--- mass on the single-token digits '1'..'4' right here (what the scorer reads): {direct:.4f}")
    print("--- full candidate strings (token ids, probability); multi-token candidates are summed over their tokens:")
    cands = digit_candidates(tok, model, ids)
    for cand, (c, total) in cands.items():
        print(f"   {cand!r:6} tokens={c} {[tok.decode([x]) for x in c]}  p={math.exp(total):.4f}")
    allmass = sum(math.exp(t) for _, t in cands.values())
    print(f"--- total probability of the answer being a bare digit or ' digit' (all 8 candidates): {allmass:.4f}")
    dl, _, _ = _digit_logprobs(tok, model, row, prompt_mode, chat, enable_thinking)
    raw = E.softmax(dl)
    print(f"--- digit probabilities renormalised over 1-4 (what the argmax sees): {[round(x, 3) for x in raw]}  -> predicts {max(range(4), key=lambda j: raw[j]) + 1}")
    if calibrate:
        prior, per_input = _prior(tok, model, row, prompt_mode, chat, enable_thinking)
        print("--- calibration: digit probabilities (renormalised) for each content-free input:")
        for t, l in zip(E.CONTENT_FREE, per_input):
            print(f"   input {t!r:8} -> {[round(x, 3) for x in E.softmax(l)]}")
        print(f"   mean log-prior -> probabilities {[round(x, 3) for x in E.softmax(prior)]}")
        cal = E.softmax([a - b for a, b in zip(dl, prior)])
        print(f"--- calibrated probabilities: {[round(x, 3) for x in cal]}  -> predicts {max(range(4), key=lambda j: cal[j]) + 1}")
    if generate:
        with torch.no_grad():
            out = model.generate(ids.to(next(model.parameters()).device), max_new_tokens=generate, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        print(f"--- greedy continuation ({generate} tokens): {tok.decode(out[0, ids.shape[1]:])!r}")


def summary(tok, model, rows, prompt_mode="nosubject", chat=False, enable_thinking=False):
    """Over many dev questions: the most common top-1 next tokens and the average digit mass."""
    digit_ids = _digit_ids(tok)
    top1 = collections.Counter()
    masses = []
    for r in rows:
        ids = _ids(tok, build_text(tok, r, prompt_mode, chat, enable_thinking), chat)
        lp = _next_logprobs(model, ids)
        top1[tok.decode([int(lp.argmax())])] += 1
        masses.append(sum(lp[i].exp().item() for i in digit_ids))
    print(f"=== chat={'on' if chat else 'off'} prompt={prompt_mode}: n={len(rows)}  mean mass on digits={sum(masses) / len(masses):.4f}")
    print("most common top-1 next tokens:")
    for t, n in top1.most_common(8):
        print(f"   {n:4d}  {t!r}")


def scoring_table(tok, model, rows, n=20, prompt_mode="nosubject", chat=True, perms=1, calibrate=False, enable_thinking=False):
    """The REAL scoring path (E.score_rows: same digit tokens, same permutation averaging, same calibration) on the first n
    dev questions. Per question: gold, prediction, 4 probabilities, digit mass, top-1 next token, and (perms > 1) the
    prediction of each option order mapped back to the original numbering."""
    import pandas as pd
    import torch
    rows = rows[:n]
    ids_per_digit = E.digit_variants(tok)
    dev = next(model.parameters()).device
    cache = {}  # prompt -> (digit log-probs, top-1 token): reused for the per-order columns

    @torch.no_grad()
    def scorer(prompts):
        out = []
        for p in prompts:
            if p not in cache:
                text = E.render_prompt(tok, p, chat, enable_thinking)
                ids = tok(text, add_special_tokens=not chat, return_tensors="pt")["input_ids"].to(dev)
                lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1).cpu()
                cache[p] = ([torch.logsumexp(lp[v], dim=0).item() for v in ids_per_digit], tok.decode([int(lp.argmax())]))
            out.append(cache[p][0])
        return out

    cfg = dict(E.DEFAULTS, prompt=prompt_mode, perms=perms, calibrate=calibrate)
    probs = E.score_rows(rows, scorer, cfg)
    orders = cyclic_perms(4)[:perms]
    recs = []
    for i, (r, p) in enumerate(zip(rows, probs)):
        pred = max(range(4), key=lambda j: p[j]) + 1
        mass, per_order = [], []
        for perm in orders:
            lps = cache[E.prompt_for(permute_options(r, perm), prompt_mode)][0]
            mass.append(sum(math.exp(x) for x in lps))
            if calibrate:
                subj = "" if prompt_mode in ("nosubject", "nosubject_strict") else r.get("subject_si", "N/A")
                pr = [sum(cache[E.prompt_for({"question": t, "choices": [t or "-"] * 4, "subject": subj, "subject_si": subj}, prompt_mode)][0][j]
                          for t in E.CONTENT_FREE) / len(E.CONTENT_FREE) for j in range(4)]
                lps = [a - b for a, b in zip(lps, pr)]
            per_order.append(perm[max(range(4), key=lambda j: lps[j])] + 1)
        top1 = cache[E.prompt_for(r, prompt_mode)][1]
        rec = {"idx": i, "subject": r["subject"], "gold": r["answer"], "pred": pred, "correct": pred == r["answer"],
               "p1": round(p[0], 3), "p2": round(p[1], 3), "p3": round(p[2], 3), "p4": round(p[3], 3),
               "digit_mass": round(sum(mass) / len(mass), 4), "top1_next_token": top1}
        if perms > 1:
            rec["pred_per_order"] = ",".join(map(str, per_order))
        recs.append(rec)
    df = pd.DataFrame(recs)
    print(f"chat={'on' if chat else 'off'} perms={perms} calibrate={calibrate} prompt={prompt_mode}: "
          f"accuracy {df.correct.mean():.2%} on the first {len(df)} questions; "
          f"mean digit mass = {cfg['digit_mass_mean']:.4f} (min {cfg['digit_mass_min']:.4f})")
    print("predicted-option counts:", df.pred.value_counts().sort_index().to_dict())
    return df


first_n_table = scoring_table  # old name
