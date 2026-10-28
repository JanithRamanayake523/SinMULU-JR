"""Codabench submission: predictions.jsonl (one line per test question, keys copied verbatim, answer as a string),
validated against the test set and zipped.

  python -m sinmmlu.submit --validate outputs/<exp_id>/predictions.jsonl
"""
import argparse
import json
import zipfile
from pathlib import Path

from .data import load_test, submission_key


def write_submission(rows, preds, out_dir, full=True):
    """full=False: rows are a subset of the test set (dev/test_subset300.jsonl); validated against their own keys."""
    out_dir = Path(out_dir)
    f = out_dir / "predictions.jsonl"
    with open(f, "w", encoding="utf-8", newline="\n") as fh:
        for r, y in zip(rows, preds):
            fh.write(json.dumps({"q_no": r["q_no"], "subject": r["subject"], "category": r["category"],
                                 "level": r["level"], "answer": str(y)}, ensure_ascii=False) + "\n")
    if not validate(f, None if full else {submission_key(r) for r in rows}):
        raise SystemExit(f"{f} failed validation; not zipped")
    with zipfile.ZipFile(out_dir / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f, "predictions.jsonl")
    print(f"wrote {out_dir / 'predictions.zip'}")


def validate(f, expected=None):
    """expected: the submission keys the file must cover exactly (default: the whole public test set)."""
    expected = expected if expected is not None else {submission_key(r) for r in load_test()}
    seen, errs = set(), []
    with open(f, encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if type(r["q_no"]) is not int:
                    errs.append(f"line {i}: q_no is not an int")
                k = submission_key(r)
                if k in seen:
                    errs.append(f"line {i}: duplicate {k}")
                seen.add(k)
                if r["answer"] not in ("1", "2", "3", "4"):
                    errs.append(f"line {i}: bad answer {r['answer']!r}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                errs.append(f"line {i}: {type(e).__name__} {e}")
    if seen != expected:
        errs.append(f"{len(expected - seen)} test keys missing, {len(seen - expected)} unknown keys")
    print("format OK" if not errs else f"{len(errs)} problems:\n" + "\n".join(errs[:20]))
    return not errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", required=True)
    validate(ap.parse_args().validate)
