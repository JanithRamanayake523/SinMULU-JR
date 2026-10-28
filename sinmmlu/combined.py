"""Global-MMLU Sinhala + sinmulu_si as one training set, with a 10% validation set (2026-10-27).

  python -m sinmmlu.combined     -> data/combined_si/competition_format_val10/{train,val}.json

Built from the two 10% splits (python -m sinmmlu.gmmlu --val-fraction 0.10, python -m sinmmlu.sinmulu
--val-fraction 0.10; made here if missing): train = both train parts, val = both val parts. So each source keeps its
own held-out questions, a combined model is validated on exactly the questions single-source models are, and the
source of a row is its q_no prefix (gsi = Global-MMLU, osi = sinmulu_si). Validation questions whose text
(norm_dedup) also occurs in the combined train part move to train, so no question text is in both.
Rows: {q_no, subject, question, choices, answer}, the fields both sources have.
"""
import collections
import json
import subprocess
import sys

from .data import ROOT, norm_dedup

SOURCES = {"gsi": (ROOT / "data" / "global_mmlu_si" / "competition_format_val10", "sinmmlu.gmmlu"),
           "osi": (ROOT / "data" / "sinmulu_si" / "competition_format_val10", "sinmmlu.sinmulu")}
OUT = ROOT / "data" / "combined_si" / "competition_format_val10"
FIELDS = ("q_no", "subject", "question", "choices", "answer")


def combine(parts):
    """parts: [(train, val), ...] -> (train, val, n_moved)."""
    train = [{k: r[k] for k in FIELDS} for tr, _ in parts for r in tr]
    val = [{k: r[k] for k in FIELDS} for _, va in parts for r in va]
    train_q = {norm_dedup(r["question"]) for r in train}
    moved = [r for r in val if norm_dedup(r["question"]) in train_q]
    ids = {r["q_no"] for r in moved}
    return train + moved, [r for r in val if r["q_no"] not in ids], len(moved)


def main():
    parts = []
    for prefix, (d, module) in SOURCES.items():
        if not (d / "train.json").exists():
            subprocess.run([sys.executable, "-m", module, "--val-fraction", "0.10"], check=True, cwd=ROOT)
        parts.append(tuple(json.load(open(d / f"{n}.json", encoding="utf-8")) for n in ("train", "val")))
    train, val, moved = combine(parts)
    assert len({r["q_no"] for r in train + val}) == len(train) + len(val), "duplicate q_no across sources"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(OUT / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"{OUT.relative_to(ROOT)}: train {len(train)} | val {len(val)} | {moved} val questions moved to train "
          f"(same text as a train question of the other source)")
    for name, rows in (("train", train), ("val", val)):
        src = collections.Counter(r["q_no"][:3] for r in rows)
        print(f"  {name}: Global-MMLU {src['gsi']}, sinmulu_si {src['osi']} | answers "
              f"{dict(sorted(collections.Counter(r['answer'] for r in rows).items()))}")


if __name__ == "__main__":
    main()
