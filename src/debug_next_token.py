"""Diagnostic helpers: what does the model actually want to output right after the prompt?
DEV DATA ONLY (labelled public train). Never run this on the public test set.

  load(model)                          -> tok, model
  show_question(tok, model, row, ...)  -> exact text fed, last tokens, top-k next tokens, digit candidates, greedy continuation
  summary(tok, model, rows, ...)       -> over many questions: what is the top-1 next token, and how much mass is on digits
"""
import collections, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import eval_logprob as E

DIGITS = E.DIGITS


def load(model_id, dtype="fp16", revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    kw = dict(revision=revision, device_map="auto" if torch.cuda.is_available() else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)  # transformers >= 5
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)
    model.eval()
    return tok, model


def _ids(tok, text, chat):
    return tok(text, add_special_tokens=not chat, return_tensors="pt")["input_ids"]


def _next_logprobs(model, ids):
    import torch
    with torch.no_grad():
        logits = model(input_ids=ids.to(next(model.parameters()).device)).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1).cpu()


def build_text(tok, row, prompt_mode="nosubject", chat=False, enable_thinking=False):
    return E.render_prompt(tok, E.prompt_for(row, prompt_mode), chat, enable_thinking)


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


def show_question(tok, model, row, prompt_mode="nosubject", chat=False, enable_thinking=False, top_k=15, generate=100):
    import torch
    text = build_text(tok, row, prompt_mode, chat, enable_thinking)
    ids = _ids(tok, text, chat)
    print(f"=== chat={'on' if chat else 'off'}  prompt={prompt_mode}  gold answer={row.get('answer')}  ({ids.shape[1]} tokens)")
    print("--- exact text fed to the model (repr, so spaces/newlines/special tokens are visible):")
    print(repr(text[-350:]))
    last = ids[0, -12:].tolist()
    print("--- last 12 input tokens:", [(i, tok.decode([i])) for i in last])
    lp = _next_logprobs(model, ids)
    top = torch.topk(lp, top_k)
    print(f"--- top-{top_k} next tokens:")
    digit_ids = {tok.encode(d, add_special_tokens=False)[0] for d in DIGITS if len(tok.encode(d, add_special_tokens=False)) == 1}
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
    if generate:
        with torch.no_grad():
            out = model.generate(ids.to(next(model.parameters()).device), max_new_tokens=generate, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        print(f"--- greedy continuation ({generate} tokens): {tok.decode(out[0, ids.shape[1]:])!r}")


def summary(tok, model, rows, prompt_mode="nosubject", chat=False, enable_thinking=False):
    """Over many dev questions: the most common top-1 next tokens and the average digit mass."""
    import torch
    digit_ids = {tok.encode(d, add_special_tokens=False)[0] for d in DIGITS if len(tok.encode(d, add_special_tokens=False)) == 1}
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


def first_n_table(tok, model, rows, n=20, prompt_mode="nosubject", chat=True, perms=1, calibrate=False, enable_thinking=False):
    """Run the REAL scoring path (same digit tokens, same score_rows aggregation as the grid) on the first n dev questions
    and show per question: gold, prediction, digit mass at the scored position, the model's top-1 next token, and the 4 probabilities."""
    import pandas as pd
    import torch
    rows = rows[:n]
    ids_per_digit = E.digit_variants(tok)
    dev = next(model.parameters()).device
    top1 = {}

    @torch.no_grad()
    def scorer(prompts):
        out = []
        for p in prompts:
            text = E.render_prompt(tok, p, chat, enable_thinking)
            ids = tok(text, add_special_tokens=not chat, return_tensors="pt")["input_ids"].to(dev)
            lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1).cpu()
            top1[p] = tok.decode([int(lp.argmax())])
            out.append([torch.logsumexp(lp[v], dim=0).item() for v in ids_per_digit])
        return out

    cfg = dict(E.DEFAULTS, prompt=prompt_mode, perms=perms, calibrate=calibrate)
    probs = E.score_rows(rows, scorer, cfg)
    first_prompts = [E.prompt_for(r, prompt_mode) for r in rows]
    recs = []
    for i, (r, p) in enumerate(zip(rows, probs)):
        pred = max(range(4), key=lambda j: p[j]) + 1
        recs.append({"idx": i, "subject": r["subject"], "gold": r["answer"], "pred": pred, "correct": pred == r["answer"],
                     "p1": round(p[0], 3), "p2": round(p[1], 3), "p3": round(p[2], 3), "p4": round(p[3], 3),
                     "top1_next_token": top1.get(first_prompts[i], "?")})
    df = pd.DataFrame(recs)
    print(f"chat={'on' if chat else 'off'} perms={perms} calibrate={calibrate} prompt={prompt_mode}: "
          f"accuracy {df.correct.mean():.2%} on the first {len(df)} questions; "
          f"mean digit mass at the scored position = {cfg['digit_mass_mean']:.4f} (min {cfg['digit_mass_min']:.4f})")
    print("predicted-option counts:", df.pred.value_counts().sort_index().to_dict())
    return df
