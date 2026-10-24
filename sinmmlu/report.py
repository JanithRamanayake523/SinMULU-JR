"""Compare finished dev experiments on the same questions (from outputs/*/preds.jsonl; no model needed).

  python -m sinmmlu.report --data dev/easy_full.jsonl --baseline <exp_id>
      table: accuracy, 95% CI, gain over the baseline, paired bootstrap 95% CI of the gain, exact McNemar p,
      Holm-adjusted p (over all compared experiments)
  python -m sinmmlu.report --data dev/easy_full.jsonl --by-subject results/subject_accuracy.csv
      accuracy per subject (one row per subject file) for every experiment; descriptive only
"""
import argparse
import csv
import json

from .data import ROOT, load_jsonl, path
from .metrics import holm, mcnemar_exact, paired_bootstrap, wilson_halfwidth


def load_experiments(data):
    """exp_id -> {uid: correct?} for every finished, non-smoke dev experiment on `data`."""
    exps = {}
    for cfg_f in sorted((ROOT / "outputs").glob("*/config.json")):
        cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
        if cfg["data"] != data or cfg["exp_id"].endswith("__smoke"):
            continue
        exps[cfg["exp_id"]] = {p["uid"]: p["pred"] == p["gold"] for p in load_jsonl(cfg_f.parent / "preds.jsonl")}
    return exps


def compare(exps, baseline):
    base = exps[baseline]
    uids = sorted(base)
    a = [base[u] for u in uids]
    others = [e for e in exps if e != baseline]
    for e in others:
        assert set(exps[e]) == set(base), f"{e} was not run on the same questions as the baseline"
    raw = [mcnemar_exact(a, [exps[e][u] for u in uids]) for e in others]
    adj = holm(raw)
    acc = sum(a) / len(a)
    print(f"{'experiment':<80} {'acc':>7} {'ci95':>6} {'gain':>7} {'gain 95% CI':>17} {'p':>7} {'p_holm':>7}")
    print(f"{baseline:<80} {acc:7.4f} {wilson_halfwidth(acc, len(a)):6.3f}   (baseline)")
    for e, p, ph in sorted(zip(others, raw, adj), key=lambda x: -sum(exps[x[0]].values())):
        b = [exps[e][u] for u in uids]
        acc_b = sum(b) / len(b)
        lo, hi = paired_bootstrap(a, b)
        print(f"{e:<80} {acc_b:7.4f} {wilson_halfwidth(acc_b, len(b)):6.3f} {acc_b - acc:+7.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] {p:7.4f} {ph:7.4f}")


def by_subject(exps, data, out):
    rows = {r["uid"]: r for r in load_jsonl(data)}
    names = sorted(exps)
    files = sorted({r["file"] for r in rows.values()})
    with open(path(out), "w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel shows Sinhala
        w = csv.writer(fh)
        w.writerow(["file", "subject_si", "n", *names])
        for f in files:
            us = [u for u, r in rows.items() if r["file"] == f]
            w.writerow([f, rows[us[0]]["subject_si"], len(us),
                        *(f"{sum(exps[e][u] for u in us) / len(us):.4f}" for e in names)])
    print(f"wrote {out} ({len(files)} subjects x {len(names)} experiments)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--by-subject", metavar="CSV")
    a = ap.parse_args()
    exps = load_experiments(a.data)
    if not exps:
        raise SystemExit(f"no finished experiments on {a.data}")
    if a.baseline:
        compare(exps, a.baseline)
    if a.by_subject:
        by_subject(exps, a.data, a.by_subject)
    if not a.baseline and not a.by_subject:
        for e, res in sorted(exps.items(), key=lambda x: -sum(x[1].values())):
            print(f"{sum(res.values()) / len(res):.4f}  {e}")


if __name__ == "__main__":
    main()
