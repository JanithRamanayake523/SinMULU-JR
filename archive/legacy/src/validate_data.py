"""Sanity checks on public train/test structure. Usage: python src/validate_data.py"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dir


def check(rows, name, need_answer):
    bad = [r for r in rows if len(r["choices"]) != 4 or (need_answer and int(r.get("answer", 0)) not in (1, 2, 3, 4))]
    keys = collections.Counter((r["level"], r["subject"], r["category"], r["q_no"]) for r in rows)
    dups = [k for k, v in keys.items() if v > 1]
    print(f"{name}: n={len(rows)} bad={len(bad)} dup_keys={len(dups)}")
    if need_answer:
        print("  answer dist:", sorted(collections.Counter(int(r['answer']) for r in rows).items()))
    print("  per level:", dict(collections.Counter(r["level"] for r in rows)))


if __name__ == "__main__":
    check(load_dir("data/public train", level="easy"), "public train", True)
    check(load_dir("data/public test"), "public test", False)
