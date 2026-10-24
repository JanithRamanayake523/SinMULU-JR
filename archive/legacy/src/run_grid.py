"""Run the zero-shot experiment grid on a dev set, skipping experiments already in results/results.csv.

Grid per model (full precision, quant=none):
  chat      : the model's DEFAULT way: chat template on if the model is chat capable (config `chat_capable`, verified on the
              model card), raw prompt otherwise (e.g. -pt / -Base models). --chat-modes both also runs chat off.
  settings  : (calibrate=False, perms=1), (calibrate=False, perms=4), (calibrate=True, perms=1)

The model is loaded ONCE per model and all pending experiments reuse it.
Usage:
  python src/run_grid.py --dry-run                         # list done / pending, load nothing
  python src/run_grid.py                                   # all three configs, full Easy dev
  python src/run_grid.py --configs configs/gemma3_1b.yaml --data dev/dev_easy_full.jsonl
"""
import argparse, csv, itertools, os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import eval_logprob as E
from data_utils import ROOT

DEFAULT_CONFIGS = ["configs/gemma3_4b.yaml", "configs/llama32_3b.yaml", "configs/qwen3_4b_instruct_2507.yaml"]   # the 1B configs stay available via --configs
SETTINGS = [(False, 1), (False, 4), (True, 1)]  # (calibrate, perms); calibrate=True is only run with perms=1


def norm_chat(v):
    """Rows now store the RESOLVED chat mode ('on'/'off'). Legacy rows: 'auto' = on (resolved by name, worked for
    Gemma/Llama); legacy 'True'/'False' came from the YAML on/off bug (chat silently OFF) -> 'invalid', never trusted."""
    v = str(v)
    if v in ("on", "off"):
        return v
    return "on" if v == "auto" else "invalid"


def done_set(data):
    """Set of (model, chat, perms, calib, quant, prompt) already recorded for this dev file."""
    f = ROOT / "results" / "results.csv"
    done = set()
    if not f.exists():
        return done
    with open(f, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"perms=(\d+) calib=(\w+) chat=(\w+) quant=(\w+)", r["method"])
            if not m or f"data={data}" not in r["notes"]:
                continue
            done.add((r["model"], norm_chat(m.group(3)), int(m.group(1)), m.group(2) == "True", m.group(4), r["prompt"]))
    return done


def plan(cfg, data, done, chat_modes="default"):
    """default: the model's own way (chat-capable -> chat on; no chat template, e.g. -pt base models -> raw prompt).
    both: chat-capable models also with chat off."""
    if chat_modes == "both":
        chats = ["on", "off"] if cfg["chat_capable"] else ["off"]
    else:
        chats = ["on"] if cfg["chat_capable"] else ["off"]
    todo, skipped = [], []
    for chat, (calib, perms) in itertools.product(chats, SETTINGS):
        key = (cfg["model"], chat, perms, calib, "none", cfg["prompt"])
        (skipped if key in done else todo).append((chat, calib, perms))
    return todo, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--data", default="dev/dev_easy_full.jsonl")
    ap.add_argument("--prompt", choices=["nosubject", "nosubject_strict", "subject_si", "subject_en"], help="override the prompt in the configs")
    ap.add_argument("--chat-modes", choices=["default", "both"], default="default",
                    help="default: each model in its own way (chat on if chat capable, else raw); both: chat-capable models also with chat off")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data_tag = Path(a.data).stem
    rows, is_test = E.load_eval_rows(a.data)
    assert not is_test, "the grid is for labelled dev sets only"
    done = done_set(a.data)
    for cpath in a.configs:
        cfg = E.load_cfg(argparse.Namespace(config=cpath, data_tag=data_tag, limit=None, **{k: None for k in E.DEFAULTS}))
        cfg["quant"] = "none"
        if a.prompt:
            cfg["prompt"] = a.prompt
        if cfg["chat_capable"] is None:
            sys.exit(f"{cpath}: set chat_capable (true/false) after checking the model card")
        todo, skipped = plan(cfg, a.data, done, a.chat_modes)
        name = Path(cpath).stem
        print(f"\n### {cfg['model']}: {len(skipped)} already done, {len(todo)} to run")
        for t in skipped:
            print("  done   chat=%s calib=%s perms=%d" % (t[0], t[1], t[2]))
        for t in todo:
            print("  TODO   chat=%s calib=%s perms=%d" % (t[0], t[1], t[2]))
        if a.dry_run or not todo:
            continue
        scorer, tok = E.make_hf_scorer(cfg)
        fert = round(E.fertility(tok, E.load_jsonl("dev/dev_easy_full.jsonl")), 3)  # all 1,851 eval questions
        print(f"[model] tokeniser fertility = {fert} tokens per Sinhala word", flush=True)
        for chat, calib, perms in todo:
            run = dict(cfg, chat=chat, calibrate=calib, perms=perms,
                       exp_id=f"{name}_{cfg['prompt']}_chat{chat}_p{perms}_c{int(calib)}_{data_tag}")
            print(f"\n=== {run['exp_id']}", flush=True)
            scorer.set_chat(chat)
            probs = E.score_rows(rows, scorer, run)
            accs, macro, share, preds, persub = E.report_dev(rows, probs, run, {})
            info = E.run_info(scorer)
            assert info["chat_resolved"] == chat, "resolved chat mode differs from the requested one"
            print(f"[run] chat={info['chat_resolved']} max_token_len={info['max_token_len']} (limit {info['token_limit']})")
            print(f"[run] probability mass on the digits 1-4: mean={run['digit_mass_mean']:.3f} min={run['digit_mass_min']:.3f}")
            if run["digit_mass_mean"] < 0.5:
                print("[run] WARNING: < 50% of the probability is on the digits 1-4; the argmax over them is weak evidence")
            E.write_outputs(run, rows, probs, False, a.data, dict(acc=accs, option_share=share, fertility_tokens_per_word=fert, **info))
            E.append_result(run, a.data, accs, len(rows), f"grid; max_len={info['max_token_len']}; fertility={fert}")
        del scorer, tok  # free GPU memory before the next model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        import subject_report
        subject_report.build(a.data)  # refresh the per-model subject CSVs after every model (survives a disconnect)
    f = ROOT / "results" / "results.csv"
    if f.exists() and not a.dry_run:
        print("\nresults.csv rows:", sum(1 for _ in open(f, encoding="utf-8")) - 1)


if __name__ == "__main__":
    main()
