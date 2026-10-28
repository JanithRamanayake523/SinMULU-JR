"""Apply the pre-registered confirmation rule (docs/log.md, 2026-10-24) and write the winner.

Rule: on the confirmation set, a candidate is adopted only if it beats the baseline by >= 1 point with Holm-corrected
exact McNemar p < 0.05 (over all candidates); the adopted candidate with the highest accuracy wins. If none is
adopted, the baseline is the winner.

  python -m sinmmlu.select_winner --data dev/archive/inference_rest.jsonl --baseline <exp_id> --candidates <exp_id> ...
Writes results/winner_<data stem>.json: winner exp_id, how to rebuild it on the test set (prompt or ensemble parts),
and the full comparison table.
"""
import argparse
import json

from .data import ROOT, path
from .metrics import holm, mcnemar_exact
from .report import load_experiments

MIN_GAIN, ALPHA = 0.01, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--fallback", help="winner if no candidate is adopted (default: the baseline)")
    ap.add_argument("--out", help="output file (default results/winner_<data stem>.json)")
    a = ap.parse_args()
    exps = load_experiments(a.data)
    missing = [e for e in [a.baseline, *a.candidates] if e not in exps]
    if missing:
        raise SystemExit(f"not finished on {a.data}: {missing}")
    uids = sorted(exps[a.baseline])
    base = [exps[a.baseline][u] for u in uids]
    acc0 = sum(base) / len(base)
    rows = []
    for e in a.candidates:
        b = [exps[e][u] for u in uids]
        rows.append({"exp_id": e, "acc": sum(b) / len(b), "gain": sum(b) / len(b) - acc0, "p": mcnemar_exact(base, b)})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        r["adopted"] = r["gain"] >= MIN_GAIN and ph < ALPHA
    adopted = sorted([r for r in rows if r["adopted"]], key=lambda r: -r["acc"])
    winner = adopted[0]["exp_id"] if adopted else (a.fallback or a.baseline)
    cfg = json.loads((ROOT / "outputs" / winner / "config.json").read_text(encoding="utf-8"))
    recipe = ({"ensemble": [json.loads((ROOT / "outputs" / p / "config.json").read_text(encoding="utf-8"))["condition"]
                            for p in cfg["parts"]], "name": winner.rsplit("__", 1)[0]}
              if cfg.get("method") == "ensemble" else {"condition": cfg["condition"]})
    print(f"baseline {a.baseline}: {acc0:.4f} (n={len(base)})")
    for r in sorted(rows, key=lambda r: -r["acc"]):
        print(f"{'ADOPTED ' if r['adopted'] else '        '}{r['exp_id']:<75} {r['acc']:.4f} {r['gain']:+.4f} "
              f"p={r['p']:.4g} p_holm={r['p_holm']:.4g}")
    print(f"WINNER {winner}")
    out = path(a.out) if a.out else ROOT / "results" / f"winner_{path(a.data).stem}.json"
    out.write_text(json.dumps({"data": a.data, "baseline": a.baseline, "baseline_acc": acc0, "winner": winner,
                               "recipe": recipe, "model_cfg": cfg.get("model_cfg", {}).get("name"), "table": rows},
                              indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
