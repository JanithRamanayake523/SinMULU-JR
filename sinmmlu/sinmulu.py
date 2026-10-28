"""sinmulu_si (our Sri Lankan MCQ collection, data/sinmulu_si/sinmulu_si.json) as LoRA training data.

  python -m sinmmlu.sinmulu                       -> data/sinmulu_si/competition_format/{train,val}.json (5%)
  python -m sinmmlu.sinmulu --val-fraction 0.10   -> data/sinmulu_si/competition_format_val10/{train,val}.json

Input: {"mcqs": [{id "osi00001", subject, question, choices (4), answer "1".."4"}]}.
  - duplicates removed: rows with the same question and options after norm_dedup (first occurrence kept)
  - validation split: 5% (or --val-fraction) of each subject (at least 1 question for subjects with >= 10), seed 42. Questions whose
    text (norm_dedup) also occurs in the other split move to train, so no question text is in both.
  - output rows: {q_no (= id), subject, question, choices, answer (int)}
Checked against the public test set by the team before use (2026-10-27).
"""
import argparse
import collections
import json

from .data import ROOT, norm_dedup, split_by_subject

SRC = ROOT / "data" / "sinmulu_si" / "sinmulu_si.json"
OUT = ROOT / "data" / "sinmulu_si" / "competition_format"
VAL_FRACTION, SEED = 0.05, 42


def prepare(mcqs, frac=VAL_FRACTION, seed=SEED):
    """Returns (train, val, n_duplicates)."""
    seen, rows = set(), []
    for r in mcqs:
        assert len(r["choices"]) == 4 and str(r["answer"]) in ("1", "2", "3", "4"), r["id"]
        k = (norm_dedup(r["question"]), tuple(norm_dedup(str(c)) for c in r["choices"]))
        if k in seen:
            continue
        seen.add(k)
        rows.append({"q_no": r["id"], "subject": r["subject"], "question": r["question"],
                     "choices": [str(c) for c in r["choices"]], "answer": int(r["answer"])})
    train, val = split_by_subject(rows, frac, seed)
    return train, val, len(mcqs) - len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    a = ap.parse_args()
    out = OUT if a.val_fraction == VAL_FRACTION else OUT.with_name(f"{OUT.name}_val{round(100 * a.val_fraction)}")
    mcqs = json.load(open(SRC, encoding="utf-8"))["mcqs"]
    train, val, dup = prepare(mcqs, a.val_fraction)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"{out.relative_to(ROOT)}: {len(mcqs)} questions | {dup} duplicates removed | train {len(train)} | val {len(val)} "
          f"({len({r['subject'] for r in val})} subjects)")
    for name, rows in (("train", train), ("val", val)):
        print(f"  {name} answers:", dict(sorted(collections.Counter(r["answer"] for r in rows).items())))


if __name__ == "__main__":
    main()
