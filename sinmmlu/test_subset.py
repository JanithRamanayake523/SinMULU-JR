"""A fixed, level-balanced subset of the public test questions (no answers): dev/test_subset300.jsonl.

  python -m sinmmlu.test_subset                 -> 100 easy + 100 medium + 100 hard
  python -m sinmmlu.evaluate --model ... --data dev/test_subset300.jsonl --prompt cot_en_role --max-new-tokens 1024
      -> outputs/<exp_id>/predictions.jsonl / .zip with the 300 answers (validated against this subset's keys)

Within each level the questions are spread over the subject files in proportion to their size (largest remainder),
then drawn with Random(seed). Rows have the same fields as the other dev files (uid, q_no, subject, category, level,
file, question, choices).
Caution: scores on test questions must not be used to choose models or settings (shared-task rule); decide on the
labelled dev sets and use this subset at most for reporting.
"""
import argparse
import collections
import random

from .data import LEVELS, load_test, write_jsonl

FIELDS = ("uid", "q_no", "subject", "category", "level", "file", "question", "choices")


def stratified(rows, per_level, seed):
    rng = random.Random(seed)
    out = []
    for lvl in LEVELS:
        by = collections.defaultdict(list)
        for r in rows:
            if r["level"] == lvl:
                by[r["file"]].append(r)
        n = sum(len(v) for v in by.values())
        quota = {f: per_level * len(v) / n for f, v in by.items()}
        take = {f: int(q) for f, q in quota.items()}
        for f in sorted(quota, key=lambda f: (-(quota[f] - take[f]), f))[:per_level - sum(take.values())]:
            take[f] += 1
        for f in sorted(by):
            out += sorted(rng.sample(by[f], take[f]), key=lambda r: r["q_no"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = stratified(load_test(), a.per_level, a.seed)
    name = f"dev/test_subset{3 * a.per_level}.jsonl"
    write_jsonl(name, [{k: r[k] for k in FIELDS} for r in rows])
    print(f"{name}: {len(rows)} questions | levels {dict(collections.Counter(r['level'] for r in rows))} | "
          f"{len({r['file'] for r in rows})} subject files")


if __name__ == "__main__":
    main()
