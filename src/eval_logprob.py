"""Zero/few-shot log-prob scoring of the option digits 1-4 after the prompt; argmax = prediction.

Works on a dev set (labelled: prints accuracy per level/subject, appends a row to results/results.csv)
or on the public test set (unlabelled: writes predictions.jsonl in the Codabench format, NO diagnostics printed
-- supervisor rule: never look at test prediction distributions).

Examples
  python src/eval_logprob.py --config configs/qwen3_0.6b.yaml --data dev/dev_easy_full.jsonl
  python src/eval_logprob.py --config configs/qwen3_0.6b.yaml --data test
  python src/eval_logprob.py --config configs/gemma3_1b.yaml --data dev/dev_easy_screen.jsonl --limit 20   # smoke test
Options (config key = CLI flag): model, revision, chat (auto|on|off), prompt (subject_si|subject_en|nosubject),
  perms (1|4), calibrate (bool), quant (none|4bit|8bit), batch_size, seed, exp_id, enable_thinking.
"""
import argparse, collections, csv, datetime, json, math, os, random, sys, zipfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (ROOT, build_prompt, cyclic_perms, load_dir, load_jsonl, norm_model, permute_options,
                        subject_si, write_jsonl)

DEFAULTS = dict(model=None, revision=None, chat="auto", prompt="subject_si", perms=1, calibrate=False, quant="none",
                batch_size=8, seed=42, exp_id=None, enable_thinking=False, dtype="auto",
                chat_capable=None)  # chat_capable: verified from the model card, NOT guessed from the name
DIGITS = ["1", "2", "3", "4"]
CONTENT_FREE = ["N/A", "", "[MASK]"]  # contextual calibration (Zhao et al. 2021)


def render_prompt(tok, prompt, chat, enable_thinking=False):
    """Exact text fed to the model: the raw prompt, or the prompt wrapped in the model's chat template."""
    if not chat:
        return prompt
    kwargs = dict(tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)
    if "enable_thinking" in (tok.chat_template or ""):  # template exposes a thinking switch (Qwen3): force it
        kwargs["enable_thinking"] = bool(enable_thinking)
    return tok.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def parse_bool(v):
    if isinstance(v, bool) or v is None:
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")
# Llama-3.x templates inject today's date into the system prompt -> would make results date-dependent.
FIXED_DATE = "26 Jul 2024"


# ----------------------------------------------------------------------------- config / data
def load_cfg(a):
    cfg = dict(DEFAULTS)
    if a.config:
        import yaml
        cfg.update({k: v for k, v in yaml.safe_load(open(ROOT / a.config, encoding="utf-8")).items() if v is not None})
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    if not cfg["model"]:
        sys.exit("model is required (config or --model)")
    # PyYAML (YAML 1.1) parses unquoted on/off as booleans -> normalise, and fail loudly on anything else
    if isinstance(cfg["chat"], bool):
        cfg["chat"] = "on" if cfg["chat"] else "off"
    assert cfg["chat"] in ("auto", "on", "off"), f"chat must be auto|on|off, got {cfg['chat']!r}"
    cfg["chat_capable"] = parse_bool(cfg["chat_capable"])
    base = cfg["exp_id"] or (Path(a.config).stem if a.config else cfg["model"].split("/")[-1])
    cfg["exp_id"] = f"{base}_{a.data_tag}" + ("_smoke" if a.limit else "")  # dev/test/smoke runs never overwrite each other
    return cfg


def load_eval_rows(data):
    """Returns (rows, is_test). Test rows keep the verbatim key fields and have no answer."""
    if data == "test":
        rows = load_dir("data/public test")
        for r in rows:
            r["subject_si"] = subject_si(r)
        return rows, True
    return load_jsonl(data), False


def prompt_for(row, mode):
    if mode == "nosubject":
        return build_prompt(row, with_subject=False)
    if mode == "subject_en":
        return build_prompt(row, subject_text=row["subject"].replace("_", " ").strip())
    return build_prompt(row)


# ----------------------------------------------------------------------------- scoring core (model independent)
def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def score_rows(rows, scorer, cfg):
    # the calibration prior depends on the subject name only if the prompt contains one
    subj_key = (lambda r: "") if cfg["prompt"] == "nosubject" else (lambda r: r.get("subject_si", "N/A"))
    """scorer(list[str prompts]) -> list of 4 log-probs (one per digit). Returns list of 4 probs in ORIGINAL option order.
    Permutation averaging: score each cyclic order, map probabilities back to the original options, average."""
    perms = cyclic_perms(4)[: int(cfg["perms"])]
    priors = {}
    if cfg["calibrate"]:
        # prior = mean log-prob over 3 content-free inputs, one prior per distinct subject name (it is in the prompt)
        for subj in sorted({subj_key(r) for r in rows}):
            dummies = [{"question": t, "choices": [t or "-"] * 4, "subject": subj, "subject_si": subj} for t in CONTENT_FREE]
            lps = scorer([prompt_for(d, cfg["prompt"]) for d in dummies])
            priors[subj] = [sum(l[j] for l in lps) / len(lps) for j in range(4)]
    probs = [[0.0] * 4 for _ in rows]
    masses = []  # total probability the model puts on the four digit tokens (before renormalising over them)
    for perm in perms:
        prompts = [prompt_for(permute_options(r, perm), cfg["prompt"]) for r in rows]
        lps = scorer(prompts)
        for i, lp in enumerate(lps):
            masses.append(sum(math.exp(x) for x in lp))
            if priors:
                lp = [a - b for a, b in zip(lp, priors[subj_key(rows[i])])]
            p = softmax(lp)
            for new_pos, orig in enumerate(perm):
                probs[i][orig] += p[new_pos] / len(perms)
    cfg["digit_mass_mean"] = sum(masses) / len(masses)
    cfg["digit_mass_min"] = min(masses)
    return probs


# ----------------------------------------------------------------------------- HF model scorer
def digit_variants(tok):
    """For each digit, all single-token spellings ('1' and ' 1'); at least one is required."""
    out = []
    for d in DIGITS:
        ids = set()
        for s in (d, " " + d):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        assert ids, f"digit {d!r} has no single-token encoding in this tokenizer"
        out.append(sorted(ids))
    return out


def make_hf_scorer(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(revision=cfg["revision"])
    if cfg["quant"] in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=cfg["quant"] == "4bit", load_in_8bit=cfg["quant"] == "8bit",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = {"auto": torch.float16 if torch.cuda.is_available() else torch.float32,
                             "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], device_map="auto" if torch.cuda.is_available() else None, **kw)
    model.eval()
    has_template = getattr(tok, "chat_template", None) is not None
    if cfg["chat_capable"] is not None and bool(cfg["chat_capable"]) != has_template:
        sys.exit(f"{cfg['model']}: config says chat_capable={cfg['chat_capable']} but tokenizer chat_template present={has_template}")
    capable = bool(cfg["chat_capable"]) if cfg["chat_capable"] is not None else has_template
    state = {"chat": False}
    ctx = getattr(model.config, "max_position_embeddings", None) or getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
    limit = ctx or 10 ** 9  # the model's own context length is the only limit
    stats = {"max_len": 0, "limit": limit, "revision": resolve_revision(cfg, model)}

    def set_chat(mode):
        """mode: exactly 'on' or 'off'. 'on' is refused for models that are not chat capable."""
        assert mode in ("on", "off"), f"chat mode must be 'on' or 'off', got {mode!r}"
        if mode == "on" and not capable:
            raise ValueError(f"{cfg['model']} is not chat capable; only chat=off is allowed")
        state["chat"] = mode == "on"
        print(f"[run] chat={mode}", flush=True)

    set_chat(("on" if capable else "off") if cfg["chat"] == "auto" else cfg["chat"])
    ids = digit_variants(tok)
    print(f"[model] {cfg['model']} revision={stats['revision']} chat_capable={capable} digit_token_ids={ids}", flush=True)

    def render(prompt):
        return render_prompt(tok, prompt, state["chat"], cfg["enable_thinking"])

    @torch.no_grad()
    def scorer(prompts):
        dev = next(model.parameters()).device
        # chat templates already contain BOS; raw prompts need it added by the tokenizer
        ids_all = tok([render(p) for p in prompts], add_special_tokens=not state["chat"])["input_ids"]
        stats["max_len"] = max(stats["max_len"], check_length(ids_all, limit))
        order = sorted(range(len(ids_all)), key=lambda i: len(ids_all[i]))  # similar lengths per batch -> little padding
        out = [None] * len(ids_all)
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            enc = tok.pad({"input_ids": [ids_all[j] for j in idx]}, padding=True, return_tensors="pt").to(dev)
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # left padding: positions start at the first real token
            try:
                logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
            except TypeError:
                logits = model(**enc, position_ids=pos).logits[:, -1].float()
            lp = torch.log_softmax(logits, dim=-1).cpu()
            for j, row in zip(idx, lp):
                out[j] = [torch.logsumexp(row[v], dim=0).item() for v in ids]
        return out

    scorer.set_chat = set_chat
    scorer.capable = capable
    scorer.stats = stats
    scorer.resolved_chat = lambda: "on" if state["chat"] else "off"
    return scorer, tok


def resolve_revision(cfg, model):
    """Exact HF commit hash of the loaded weights (transformers 5 no longer exposes config._commit_hash)."""
    rev = getattr(model.config, "_commit_hash", None)
    if rev:
        return rev
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(cfg["model"], revision=cfg["revision"]).sha
    except Exception:
        return None


def check_length(ids_all, limit):
    """Longest prompt in tokens; raises instead of truncating (truncating would cut off the answer position)."""
    longest = max(len(x) for x in ids_all)
    if longest > limit:
        raise ValueError(f"prompt of {longest} tokens exceeds the model context length {limit}; never truncate an MCQ prompt")
    return longest


def env_info():
    import platform
    info = {"python": platform.python_version()}
    try:
        import torch, transformers
        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        import accelerate
        info["accelerate"] = accelerate.__version__
    except Exception:
        pass
    return info


def run_info(scorer):
    """Fields stored with every run: the RESOLVED chat mode, longest prompt, model revision, library versions."""
    return {"chat_resolved": scorer.resolved_chat(), "max_token_len": scorer.stats["max_len"],
            "token_limit": scorer.stats["limit"], "model_revision_resolved": scorer.stats["revision"], **env_info()}


def fertility(tok, rows, n=None):
    """Average tokens per whitespace-separated Sinhala word (lower = tokenizer knows Sinhala better)."""
    rows = rows[:n] if n else rows
    words = sum(len(norm_model(r["question"]).split()) for r in rows)
    toks = sum(len(tok.encode(norm_model(r["question"]), add_special_tokens=False)) for r in rows)
    return toks / max(words, 1)


# ----------------------------------------------------------------------------- outputs
def wilson(acc, n, z=1.96):
    if n == 0:
        return 0.0
    c = (acc + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return h


def report_dev(rows, probs, cfg, extra):
    preds = [max(range(4), key=lambda j: p[j]) + 1 for p in probs]
    by = collections.defaultdict(list)
    for r, p in zip(rows, preds):
        by[r["level"]].append(p == int(r["answer"]))
    accs = {lvl: sum(v) / len(v) for lvl, v in by.items()}
    for lvl, v in sorted(by.items()):
        a = accs[lvl]
        print(f"{lvl}: acc={a:.4f} n={len(v)} (95% CI +-{wilson(a, len(v)):.3f})")
    macro = sum(accs.get(l, float("nan")) for l in ("easy", "medium", "hard") if l in accs) / len(accs)
    hist = collections.Counter(preds)
    share = {k: hist.get(k, 0) / len(preds) for k in (1, 2, 3, 4)}
    print("predicted-option share:", {k: round(v, 3) for k, v in share.items()},
          "<- BIAS (>35%)" if max(share.values()) > 0.35 else "")
    persub = collections.defaultdict(list)
    for r, p in zip(rows, preds):
        persub[r.get("subject_si") or r["subject"]].append(p == int(r["answer"]))
    return accs, macro, share, preds, {s: (sum(v) / len(v), len(v)) for s, v in persub.items()}


def write_outputs(cfg, rows, probs, is_test, data, extra):
    out = ROOT / "outputs" / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "data": data, **extra}, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = [(r["level"], r["subject"], r["category"], r["q_no"]) for r in rows]
    with open(out / "probs.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for k, p in zip(keys, probs):
            fh.write(json.dumps({"key": list(k), "p": [float(x) for x in p]}) + "\n")
    if is_test:
        # keys copied VERBATIM from the test row; answer is a STRING
        with open(out / "predictions.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r, p in zip(rows, probs):
                fh.write(json.dumps({"q_no": r["q_no"], "subject": r["subject"], "category": r["category"],
                                     "level": r["level"], "answer": str(max(range(4), key=lambda j: p[j]) + 1)},
                                    ensure_ascii=False) + "\n")
        from predict import validate
        ok = validate(str(out / "predictions.jsonl"))
        if ok:
            with zipfile.ZipFile(out / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.write(out / "predictions.jsonl", "predictions.jsonl")
        print(f"wrote {out / 'predictions.jsonl'} (format valid: {ok})")
    return out


def append_result(cfg, data, accs, n_eval, notes):
    """results.csv has `dev_easy` (our eval set is Easy-only). Any other level in a dev file goes into notes."""
    f = ROOT / "results" / "results.csv"
    cols = ["exp_id", "date", "model", "method", "prompt", "dev_easy", "ci95", "digit_mass", "notes"]
    new = not f.exists()
    if not new and open(f, encoding="utf-8").readline().strip() != ",".join(cols):
        sys.exit("results/results.csv has an outdated header: run the fresh-start cell (or add the digit_mass column) first")
    other = " ".join(f"{l}={v:.4f}" for l, v in accs.items() if l != "easy")
    easy = accs.get("easy")
    with open(f, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        method = f"zeroshot_logprob perms={cfg['perms']} calib={cfg['calibrate']} chat={cfg['chat']} quant={cfg['quant']}"
        w.writerow([cfg["exp_id"], datetime.date.today().isoformat(), cfg["model"], method, cfg["prompt"],
                    f"{easy:.4f}" if easy is not None else "", f"{wilson(easy, n_eval):.4f}" if easy is not None else "",
                    f"{cfg['digit_mass_mean']:.3f}" if "digit_mass_mean" in cfg else "",
                    f"data={data}; {notes}" + (f"; {other}" if other else "")])


def length_report(cfg, rows):
    """Token-length check without a model weights download: how long are the prompts vs the model's context length."""
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    mc = AutoConfig.from_pretrained(cfg["model"], revision=cfg["revision"])
    ctx = getattr(mc, "max_position_embeddings", None) or getattr(getattr(mc, "text_config", None), "max_position_embeddings", None)
    for mode in ("off", "on"):
        if mode == "on" and not getattr(tok, "chat_template", None):
            continue
        ls = []
        for r in rows:
            p = prompt_for(r, cfg["prompt"])
            if mode == "on":
                p = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)
            ls.append(len(tok(p, add_special_tokens=(mode == "off"))["input_ids"]))
        ls.sort()
        print(f"{cfg['model']} chat={mode}: n={len(ls)} max={ls[-1]} p99={ls[int(.99 * len(ls))]} model_context={ctx} "
              f"over_context={sum(l > ctx for l in ls) if ctx else '?'} (info: over_1024={sum(l > 1024 for l in ls)})")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data", required=True, help="dev jsonl path, or the word 'test' for the public test set")
    ap.add_argument("--limit", type=int, help="smoke test: first N rows only (never for real results)")
    ap.add_argument("--fertility", action="store_true", help="also report tokeniser fertility on 1000 train questions")
    ap.add_argument("--no-results-row", action="store_true")
    ap.add_argument("--lengths-only", action="store_true", help="tokenizer only: report prompt token lengths, no model")
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if isinstance(v, bool) or k == "chat_capable":
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=parse_bool)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t)
    a = ap.parse_args()
    a.data_tag = "test" if a.data == "test" else Path(a.data).stem
    cfg = load_cfg(a)
    rows, is_test = load_eval_rows(a.data)
    if a.limit:
        rows = rows[: a.limit]
    if a.lengths_only:
        return length_report(cfg, rows)
    scorer, tok = make_hf_scorer(cfg)
    cfg["chat"] = scorer.resolved_chat()  # store the RESOLVED value, never the config value
    extra = {}
    if a.fertility:
        extra["fertility_tokens_per_word"] = round(fertility(tok, load_jsonl("dev/dev_easy_full.jsonl")), 3)
        print("fertility:", extra["fertility_tokens_per_word"])
    probs = score_rows(rows, scorer, cfg)
    extra.update(run_info(scorer))
    print(f"[run] chat={extra['chat_resolved']} max_token_len={extra['max_token_len']} (limit {extra['token_limit']})")
    print(f"[run] probability mass on the digits 1-4: mean={cfg['digit_mass_mean']:.3f} min={cfg['digit_mass_min']:.3f}")
    if cfg["digit_mass_mean"] < 0.5:
        print("[run] WARNING: the model puts < 50% of its probability on the digits 1-4; the argmax over them is weak evidence (check prompt / chat template)")
    if is_test:
        write_outputs(cfg, rows, probs, True, a.data, extra)
    else:
        accs, macro, share, preds, persub = report_dev(rows, probs, cfg, extra)
        extra.update(acc=accs, macro=macro, option_share=share)
        write_outputs(cfg, rows, probs, False, a.data, extra)
        if not a.limit and not a.no_results_row:
            append_result(cfg, a.data, accs, len(rows), f"fertility={extra.get('fertility_tokens_per_word', '')}; max_len={extra['max_token_len']}")


if __name__ == "__main__":
    main()
