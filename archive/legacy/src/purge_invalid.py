"""Quarantine results produced with the YAML `chat: on` bug (chat silently OFF, recorded as chat=True/False).
Moves their outputs/ folders to outputs/_invalid_chat_bug/ and their rows to results/invalid_rows.csv, then removes
them from results.csv so run_grid.py reruns them correctly.
--models SUBSTR ...  additionally quarantines ALL rows of those models (use when --lengths-only shows that prompts of
that model exceeded the old 1024-token truncation limit, i.e. their old rows may contain truncated prompts).
Usage: python src/purge_invalid.py [--models llama] [--apply]     (default: dry run)"""
import argparse, csv, os, re, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--models", nargs="*", default=[], help="substrings of model ids whose rows are quarantined too")
    a = ap.parse_args()
    f = ROOT / "results" / "results.csv"
    rows = list(csv.DictReader(open(f, encoding="utf-8", newline="")))
    bad = [r for r in rows if re.search(r"chat=(True|False)\b", r["method"])
           or any(m.lower() in r["model"].lower() for m in a.models)]
    print(f"{len(bad)} invalid rows of {len(rows)}:", [r["exp_id"] for r in bad])
    if not a.apply or not bad:
        return
    cols = list(rows[0].keys())
    keep = [r for r in rows if r not in bad]
    with open(ROOT / "results" / "invalid_rows.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(bad)
    with open(f, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(keep)
    q = ROOT / "outputs" / "_invalid_chat_bug"
    q.mkdir(parents=True, exist_ok=True)
    for r in bad:
        src = ROOT / "outputs" / r["exp_id"]
        if src.exists():
            shutil.move(str(src), str(q / r["exp_id"]))
    print("quarantined; results.csv now has", len(keep), "rows")


if __name__ == "__main__":
    main()
