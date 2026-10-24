"""Run one model under one or more conditions on a dev file or on the public test set.

A condition = (prompt, chat, perms, calibrate, translator). The prompt decides the scoring method (prompts.py):
  digits : next-token probability of the 4 option labels (option-order averaging and calibration apply)
  cloze  : length-normalised log-likelihood of each option text
  cot    : greedy reasoning first, then the label probabilities after the answer cue
The model is loaded once; every condition is one experiment:
  outputs/<exp_id>/config.json        model config, condition, model info (revision, n_params, dtype), environment, stats
  outputs/<exp_id>/preds.jsonl        uid, probs (original option order), pred (+ gold on dev)
  outputs/<exp_id>/generations.jsonl  cot only: the reasoning text per question
  outputs/<exp_id>/predictions.zip    test only (Codabench format, validated)
Dev runs append a row to results/results.csv. Test runs print no accuracy-like diagnostics. Finished experiments
are skipped unless --force.

Examples
  python -m sinmmlu.evaluate --model configs/models/gemma3_4b_it.yaml --data dev/inference_subset.jsonl \\
      --prompt paper_nosubject paper_subject_si --perms 1 4 --calibrate false true          # all combinations
  python -m sinmmlu.evaluate --model ... --data dev/inference_subset.jsonl --prompt en_nosubject --translator nllb
  python -m sinmmlu.evaluate --model ... --data test --prompt paper_subject_si --perms 4
  python -m sinmmlu.evaluate --model ... --data dev/inference_subset.jsonl --limit 8      # smoke test, no results row
"""
import argparse
import collections
import csv
import datetime
import itertools
import json
import math
import re

import yaml

from . import prompts as P
from .data import LEVELS, ROOT, load_jsonl, load_rows, norm_model, path
from .metrics import wilson_halfwidth
from .scoring import argmax1, score_rows, softmax

RESULTS = ROOT / "results" / "results.csv"
COLUMNS = ["exp_id", "date", "model", "quant", "dtype", "method", "prompt", "chat", "perms", "calibrate", "translator",
           "data", "n", "acc_easy", "acc_medium", "acc_hard", "ci95", "label_mass", "max_option_share", "parse_rate",
           "max_len", "seconds", "git_rev"]
ANSWER_RE = re.compile(r"(?:පිළිතුර|Answer)\s*[:：]?\s*\**\s*([1-4])")


def parse_bool(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {v!r}")


def load_model_cfg(p):
    cfg = yaml.safe_load(open(path(p), encoding="utf-8"))
    for k in ("name", "model", "chat_capable"):
        assert k in cfg, f"{p}: missing {k}"
    for k, v in dict(revision=None, dtype="auto", quant="none", batch_size=8, gen_batch_size=16, seed=42).items():
        cfg.setdefault(k, v)
    assert isinstance(cfg["chat_capable"], bool), f"{p}: chat_capable must be true/false"
    return cfg


def exp_id(mcfg, cond, data):
    tag = "test" if data == "test" else path(data).stem
    extra = ""  # reasoning options, only when set (keeps the ids of earlier experiments unchanged)
    if cond.get("max_new_tokens"):
        extra += f"__mt{cond['max_new_tokens']}"
    if cond.get("samples", 1) > 1:
        extra += f"__sc{cond['samples']}t{cond.get('temperature', 0.7)}"
    return (f"{mcfg['name']}__{mcfg['quant']}__{cond['prompt']}__chat{cond['chat']}__p{cond['perms']}"
            f"__c{int(cond['calibrate'])}__tr{cond['translator']}{extra}__{tag}")


def translator_name(cond, mcfg):
    from .translate import backend_name
    return backend_name(cond["translator"], mcfg)


def check_condition(c, mcfg):
    p = P.PROMPTS[c["prompt"]]
    if c["chat"] == "on" and not mcfg["chat_capable"]:
        raise SystemExit(f"{mcfg['model']} is not chat capable")
    if p.method in ("cloze", "cot") and (c["perms"] != 1 or c["calibrate"]):
        raise SystemExit(f"{c['prompt']}: perms/calibration only apply to digit scoring")
    if p.method == "cot" and c["chat"] != "on":
        raise SystemExit(f"{c['prompt']}: reasoning prompts need the chat template")
    if (p.lang == "si") != (c["translator"] == "none"):
        raise SystemExit(f"{c['prompt']} (lang={p.lang}) with translator={c['translator']}: "
                         "translated prompts need a translator, Sinhala-only prompts must not have one")
    if c["translator"] == "self" and not mcfg["chat_capable"]:
        raise SystemExit("self-translation needs a chat model")
    if p.method != "cot" and (c.get("max_new_tokens") or c.get("samples", 1) > 1):
        raise SystemExit(f"{c['prompt']}: --max-new-tokens / --samples only apply to reasoning prompts")


# ----------------------------------------------------------------------------- methods
def run_cloze(lm, rows, prompt):
    res = lm.cloze([P.build(r, prompt) for r in rows], [[norm_model(c) for c in r["choices"]] for r in rows],
                   P.PROMPTS[prompt].answer_word)
    probs = [softmax([s / n for s, n in opt]) for opt in res]
    return probs, {}


def parse_answer(text):
    m = list(ANSWER_RE.finditer(text))
    return (int(m[-1].group(1)), m[-1].start()) if m else (None, None)


def generate_cached(lm, rows, texts, gen_file, max_new, sample_seed=None, temperature=0.7):
    """Reasoning text per row, cached in gen_file (an interrupted run resumes)."""
    cached = {g["uid"]: g["text"] for g in load_jsonl(gen_file)} if gen_file.exists() else {}
    todo = [i for i, r in enumerate(rows) if r["uid"] not in cached]
    if todo:
        gens = lm.generate([texts[i] for i in todo], max_new, rendered=True, sample_seed=sample_seed,
                           temperature=temperature)
        with open(gen_file, "a", encoding="utf-8", newline="\n") as fh:
            for i, g in zip(todo, gens):
                cached[rows[i]["uid"]] = g
                fh.write(json.dumps({"uid": rows[i]["uid"], "text": g}, ensure_ascii=False) + "\n")
    return [cached[r["uid"]] for r in rows]


def run_cot(lm, rows, prompt, out_dir, cond=None, seed=42):
    """Greedy reasoning, or self-consistency (cond samples=k > 1): k sampled reasoning texts per question
    (temperature cond temperature, top-p 0.95, seeds seed+1000*i), each scored after the cue; the k label
    distributions are averaged (soft vote). stats also report the hard majority vote of the stated answers."""
    cond = cond or {}
    p = P.PROMPTS[prompt]
    max_new = cond.get("max_new_tokens") or p.max_new_tokens
    k = cond.get("samples", 1)
    texts = [lm.render(P.build(r, prompt)) for r in rows]
    probs = [[0.0] * 4 for _ in rows]
    parsed_all, masses, agree, n_parsed = [], [], 0, 0
    for s in range(k):
        if k == 1:
            gens = generate_cached(lm, rows, texts, out_dir / "generations.jsonl", max_new)
        else:
            gens = generate_cached(lm, rows, texts, out_dir / f"generations_s{s}.jsonl", max_new,
                                   sample_seed=seed + 1000 * s, temperature=cond.get("temperature", 0.7))
        scored, parsed = [], []
        for t, g in zip(texts, gens):
            ans, start = parse_answer(g)
            parsed.append(ans)
            reasoning = g[:start] if start is not None else g  # cut the stated answer; the cue asks for it again
            scored.append(t + reasoning.rstrip() + p.cue)
        lps = lm.label_logprobs_text(scored, p.labels)
        for i, lp in enumerate(lps):
            q = softmax(lp)
            probs[i] = [a + b / k for a, b in zip(probs[i], q)]
            masses.append(sum(math.exp(x) for x in lp))
            if parsed[i] is not None:
                n_parsed += 1
                agree += parsed[i] == argmax1(q)
        parsed_all.append(parsed)
    stats = {"parse_rate": n_parsed / (len(rows) * k), "parsed_agrees_with_scored": agree / max(n_parsed, 1),
             "label_mass_mean": sum(masses) / len(masses), "label_mass_min": min(masses),
             "max_new_tokens": max_new, "samples": k}
    if k > 1:
        votes = []
        for i in range(len(rows)):
            c = collections.Counter(a for a in (pa[i] for pa in parsed_all) if a is not None)
            votes.append(max(c, key=lambda a: (c[a], probs[i][a - 1])) if c else argmax1(probs[i]))
        stats["majority_vote_preds"] = votes
    return probs, stats


# ----------------------------------------------------------------------------- bookkeeping
def summarise(rows, preds):
    by = collections.defaultdict(list)
    for r, p in zip(rows, preds):
        by[r["level"]].append(p == int(r["answer"]))
    accs = {lvl: sum(v) / len(v) for lvl, v in by.items()}
    share = collections.Counter(preds)
    return accs, {k: share.get(k, 0) / len(preds) for k in (1, 2, 3, 4)}


def append_result(row):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    if not new:
        with open(RESULTS, encoding="utf-8") as fh:
            assert fh.readline().strip() == ",".join(COLUMNS), "results.csv header differs from COLUMNS"
    with open(RESULTS, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def save_experiment(eid, data, rows, probs, stats, cfg_extra, limit=None, write_row=True):
    """Shared by evaluate and ensemble: preds.jsonl, config.json, submission (test) or results row (dev)."""
    from .submit import write_submission
    out = ROOT / "outputs" / eid
    out.mkdir(parents=True, exist_ok=True)
    preds = [argmax1(p) for p in probs]
    is_test = data == "test"
    with open(out / "preds.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r, p, y in zip(rows, probs, preds):
            rec = {"uid": r["uid"], "probs": [round(x, 6) for x in p], "pred": y}
            if not is_test:
                rec["gold"] = int(r["answer"])
            fh.write(json.dumps(rec) + "\n")
    votes = stats.pop("majority_vote_preds", None)  # self-consistency: hard vote of the stated answers
    if votes and not is_test:
        stats["majority_vote_acc"] = sum(v == int(r["answer"]) for v, r in zip(votes, rows)) / len(rows)
    summary = {}
    if is_test:
        write_submission(rows, preds, out)
    else:
        accs, share = summarise(rows, preds)
        summary = {"acc": accs, "option_share": share}
    (out / "config.json").write_text(json.dumps({"exp_id": eid, "data": data, "n": len(rows), **cfg_extra,
                                                 "stats": stats, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    msg = f"[{eid}] n={len(rows)}"
    if "label_mass_mean" in stats:
        msg += f" label_mass={stats['label_mass_mean']:.3f} (min {stats['label_mass_min']:.3f})"
        if stats["label_mass_mean"] < 0.5:
            msg += " WARNING: < 50% of the probability is on the labels"
    if "parse_rate" in stats:
        msg += f" parse_rate={stats['parse_rate']:.3f}"
    if "majority_vote_acc" in stats:
        msg += f" majority_vote_acc={stats['majority_vote_acc']:.4f}"
    if not is_test:
        accs, share = summary["acc"], summary["option_share"]
        msg += "  " + " ".join(f"{l}={accs[l]:.4f}" for l in LEVELS if l in accs)
        msg += f"  option_share={[round(share[k], 3) for k in (1, 2, 3, 4)]}"
        if write_row and not limit:
            n_e = sum(r["level"] == "easy" for r in rows)
            c, m = cfg_extra.get("condition", {}), cfg_extra.get("model_cfg", {})
            append_result({
                "exp_id": eid, "date": datetime.date.today().isoformat(), "model": m.get("model", "ensemble"),
                "quant": m.get("quant", ""), "dtype": cfg_extra.get("model_info", {}).get("dtype_resolved", ""),
                "method": cfg_extra.get("method", ""), "prompt": c.get("prompt", ""), "chat": c.get("chat", ""),
                "perms": c.get("perms", ""), "calibrate": c.get("calibrate", ""), "translator": c.get("translator", ""),
                "data": data, "n": len(rows), **{f"acc_{l}": f"{accs[l]:.4f}" if l in accs else "" for l in LEVELS},
                "ci95": f"{wilson_halfwidth(accs['easy'], n_e):.4f}" if "easy" in accs else "",
                "label_mass": f"{stats['label_mass_mean']:.3f}" if "label_mass_mean" in stats else "",
                "max_option_share": f"{max(share.values()):.3f}",
                "parse_rate": f"{stats['parse_rate']:.3f}" if "parse_rate" in stats else "",
                "max_len": stats.get("max_len", ""), "seconds": stats.get("seconds", ""),
                "git_rev": cfg_extra.get("env", {}).get("git_rev")})
    print(msg, flush=True)
    return out


def run_condition(model, tok, minfo, mcfg, cond, rows, data, limit, env, force=False):
    import time
    from .model import LM
    from .translate import attach
    eid = exp_id(mcfg, cond, data) + ("__smoke" if limit else "")
    out = ROOT / "outputs" / eid
    out.mkdir(parents=True, exist_ok=True)
    if force:
        for f in out.glob("generations*.jsonl"):
            f.unlink()  # otherwise reasoning from an interrupted run is reused
    p = P.PROMPTS[cond["prompt"]]
    if cond["translator"] != "none":
        rows = attach(rows, translator_name(cond, mcfg))
    lm = LM(model, tok, chat=cond["chat"] == "on", batch_size=mcfg["batch_size"], context=minfo["context_length"],
            gen_batch_size=mcfg["gen_batch_size"])
    t0 = time.time()
    if p.method == "digits":
        probs, stats = score_rows(rows, lm, cond["prompt"], cond["perms"], cond["calibrate"])
    elif p.method == "cloze":
        probs, stats = run_cloze(lm, rows, cond["prompt"])
    else:
        probs, stats = run_cot(lm, rows, cond["prompt"], out, cond, mcfg["seed"])
    stats.update(max_len=lm.max_len, seconds=round(time.time() - t0, 1))
    save_experiment(eid, data, rows, probs, stats,
                    {"method": p.method, "model_cfg": mcfg, "condition": cond, "model_info": minfo, "env": env,
                     "translator_meta": translator_meta(cond, mcfg)}, limit)


def translator_meta(cond, mcfg):
    if cond["translator"] == "none":
        return None
    from .translate import CACHE
    f = CACHE / f"{translator_name(cond, mcfg)}.meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="configs/models/<name>.yaml")
    ap.add_argument("--data", required=True, help="dev .jsonl, or 'test'")
    ap.add_argument("--prompt", nargs="+", default=["paper_nosubject"], choices=sorted(P.PROMPTS))
    ap.add_argument("--chat", nargs="+", default=["auto"], choices=["auto", "on", "off"],
                    help="auto = on if the model is chat capable")
    ap.add_argument("--perms", nargs="+", type=int, default=[1], choices=[1, 2, 3, 4])
    ap.add_argument("--calibrate", nargs="+", type=parse_bool, default=[False])
    ap.add_argument("--translator", nargs="+", default=["none"], choices=["none", "nllb", "madlad", "self"])
    ap.add_argument("--max-new-tokens", nargs="+", type=int, default=[0],
                    help="reasoning prompts: override the reasoning length limit (0 = the prompt's own)")
    ap.add_argument("--samples", nargs="+", type=int, default=[1],
                    help="reasoning prompts: self-consistency with k sampled reasoning texts (1 = greedy)")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for --samples > 1")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], help="override the model config")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp32", "fp16"], help="override the model config")
    ap.add_argument("--limit", type=int, help="first N rows only (smoke test; no results row)")
    ap.add_argument("--force", action="store_true", help="rerun finished experiments")
    a = ap.parse_args()

    mcfg = load_model_cfg(a.model)
    for k in ("quant", "dtype"):
        if getattr(a, k):
            mcfg[k] = getattr(a, k)
    rows = load_rows(a.data)
    if a.limit:
        rows = rows[: a.limit]
    conds = []
    for prompt, chat, perms, calib, tr, mt, k in itertools.product(a.prompt, a.chat, a.perms, a.calibrate, a.translator,
                                                                    a.max_new_tokens, a.samples):
        chat = ("on" if mcfg["chat_capable"] else "off") if chat == "auto" else chat
        c = {"prompt": prompt, "chat": chat, "perms": perms, "calibrate": calib, "translator": tr}
        if mt:
            c["max_new_tokens"] = mt
        if k > 1:
            c.update(samples=k, temperature=a.temperature)
        check_condition(c, mcfg)
        if c not in conds:
            conds.append(c)
    todo = []
    for c in conds:
        eid = exp_id(mcfg, c, a.data) + ("__smoke" if a.limit else "")
        done = (ROOT / "outputs" / eid / "config.json").exists()
        print(("skip   " if done and not a.force else "run    ") + eid)
        if a.force or not done:
            todo.append(c)
    if not todo:
        return
    from .model import env_info, load_lm, seed_everything
    seed_everything(mcfg["seed"])
    model, tok, minfo = load_lm(mcfg["model"], mcfg["revision"], mcfg["dtype"], mcfg["quant"])
    print(f"[model] {mcfg['model']} {json.dumps(minfo)}", flush=True)
    env = env_info()
    for i, c in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {exp_id(mcfg, c, a.data)}", flush=True)
        run_condition(model, tok, minfo, mcfg, c, rows, a.data, a.limit, env, a.force)


if __name__ == "__main__":
    main()
