"""Per-model subject-level (= per source JSON file) accuracy on the Easy eval set, built from saved probs.jsonl.
No model is run. One CSV per model: results/subject_accuracy/<ModelName>.csv
  rows    : one per JSON file (subject) + an ALL row
  columns : file, subject, n, then one accuracy column per experiment (chat / perms / calibration)
Usage: python src/subject_report.py [--data dev/dev_easy_full.jsonl]"""
import argparse, collections, csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import ROOT, load_jsonl

def chat_mode(cfg):
    """Resolved mode if stored; legacy: 'auto' = on; True/False came from the YAML on/off bug -> None (skipped)."""
    v = cfg.get("chat_resolved") or cfg.get("chat")
    return v if v in ("on", "off") else ("on" if v == "auto" else None)

def label(cfg):
    return f"{cfg.get('prompt', 'subject_si')}_chat-{chat_mode(cfg)}_perms{cfg['perms']}_calib{int(bool(cfg['calibrate']))}"

def build(data):
    rows = load_jsonl(data)
    key2 = {(r["level"], r["subject"], r["category"], r["q_no"]): r for r in rows}
    per_model = collections.defaultdict(dict)       # model -> label -> {file: [correct list]}
    for cfgf in sorted((ROOT / "outputs").glob("*/config.json")):
        cfg = json.loads(cfgf.read_text(encoding="utf-8"))
        if cfg.get("data") != data or cfg["exp_id"].endswith("_smoke") or chat_mode(cfg) is None:
            continue
        res = collections.defaultdict(list)
        for l in open(cfgf.parent / "probs.jsonl", encoding="utf-8"):
            d = json.loads(l)
            r = key2[tuple(d["key"])]
            res[r["file"]].append(max(range(4), key=lambda j: d["p"][j]) + 1 == r["answer"])
        per_model[cfg["model"].split("/")[-1]][label(cfg)] = res
    meta = {}
    for r in rows:
        meta.setdefault(r["file"], r["subject_si"])
    nfile = collections.Counter(r["file"] for r in rows)
    out_dir = ROOT / "results" / "subject_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, exps in per_model.items():
        labels = sorted(exps)
        with open(out_dir / f"{model}.csv", "w", encoding="utf-8-sig", newline="") as fh:   # BOM so Excel shows Sinhala
            w = csv.writer(fh)
            w.writerow(["file", "subject", "n", *labels])
            for f in sorted(nfile):
                w.writerow([f, meta[f], nfile[f], *(f"{sum(exps[l][f]) / len(exps[l][f]):.4f}" for l in labels)])
            w.writerow(["ALL", "", len(rows), *(f"{sum(sum(v) for v in exps[l].values()) / len(rows):.4f}" for l in labels)])
        print(f"{model}: {len(labels)} experiments -> {out_dir / (model + '.csv')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="dev/dev_easy_full.jsonl")
    build(ap.parse_args().data)
