"""Validate predictions.jsonl and zip it. Usage: python src/predict.py --validate predictions.jsonl
Keys (q_no, subject, category, level) must be copied VERBATIM from the test rows (no normalisation).
Model inference hook is TODO (section 5)."""
import argparse, json, os, sys, zipfile
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dir


def key(r):
    return (r["level"], r["subject"], r["category"], r["q_no"])


def validate(path, test_dir="data/public test"):
    expected = {key(r) for r in load_dir(test_dir)}
    seen, errs = set(), []
    with open(path, encoding="utf-8-sig") as fh:  # tolerate a BOM (PowerShell default output)
        for i, l in enumerate(fh, 1):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                if type(r["q_no"]) is not int:
                    errs.append(f"line {i}: q_no is not an int")
                k = key(r)
                if k in seen:
                    errs.append(f"line {i}: duplicate {k}")
                seen.add(k)
                if r["answer"] not in ("1", "2", "3", "4"):
                    errs.append(f"line {i}: bad answer {r['answer']!r}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                errs.append(f"line {i}: {type(e).__name__} {e}")
    if len(seen) != len(expected):
        errs.append(f"{len(seen)} unique rows, expected {len(expected)}")
    if seen != expected:
        errs.append(f"keys differ from test set: {len(expected - seen)} missing, {len(seen - expected)} unknown")
    print("OK" if not errs else f"{len(errs)} problems:\n" + "\n".join(errs[:20]))
    return not errs


def zip_it(path, out="predictions.zip"):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "predictions.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", required=True)
    a = ap.parse_args()
    if validate(a.validate):
        zip_it(a.validate)
