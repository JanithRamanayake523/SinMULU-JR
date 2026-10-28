"""Global-MMLU Sinhala (CohereLabs/Global-MMLU, config si) as training data in the competition format.

  python -m sinmmlu.gmmlu                       -> data/global_mmlu_si/competition_format/global_mmlu_si.json
  python -m sinmmlu.gmmlu --val-fraction 0.10   -> also data/global_mmlu_si/competition_format_val10/{train,val}.json:
      10% of each subject as validation (seed 42, as for sinmulu_si); the validation set for Global-MMLU-only models

Downloads the two parquet files of a fixed dataset revision (test 14,042 + dev 285 questions; Apache-2.0) and writes
one JSON list of {q_no, subject, category, question, choices, answer}:
  q_no      "gsi00001".. in source order (test split first, then dev: gsi14040.. are the dev questions)
  subject   MMLU subject name, e.g. "Abstract Algebra"
  category  competition names: stem, humanities, social_science, business; Medical and Other -> other
  answer    1-4 (A-D)
Questions with an empty question or option are dropped (3 in this revision, all with an empty option A).
External data: never used for evaluation; checked for near-duplicates of the public test set (none, 2026-10-27).
"""
import argparse
import collections
import json

from .data import ROOT, split_by_subject

SEED = 42
REPO, REVISION = "CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
FILES = (("test", "si/test-00000-of-00001.parquet"), ("dev", "si/dev-00000-of-00001.parquet"))
OUT = ROOT / "data" / "global_mmlu_si" / "competition_format" / "global_mmlu_si.json"
CATEGORY = {"STEM": "stem", "Humanities": "humanities", "Social Sciences": "social_science",
            "Business": "business", "Medical": "other", "Other": "other"}


def convert(source_rows):
    """source_rows: Global-MMLU rows, test split first. Returns (rows in competition format, dropped sample ids)."""
    out, dropped = [], []
    for r in source_rows:
        choices = [r[f"option_{c}"] for c in "abcd"]
        if not (r["question"] or "").strip() or any(c is None or not str(c).strip() for c in choices):
            dropped.append(r["sample_id"])
            continue
        out.append({"q_no": f"gsi{len(out) + 1:05d}", "subject": r["subject"].replace("_", " ").title(),
                    "category": CATEGORY[r["subject_category"]], "question": r["question"],
                    "choices": [str(c) for c in choices], "answer": "ABCD".index(r["answer"]) + 1})
    return out, dropped


def write_split(frac):
    """Validation split of OUT for Global-MMLU-only training: frac of each subject (data.split_by_subject, seed 42),
    instead of the 285 Global-MMLU dev questions (q_no >= gsi14040) that adapter A held out."""
    rows = json.load(open(OUT, encoding="utf-8"))
    train, val = split_by_subject(rows, frac, SEED)
    d = OUT.parent.with_name(f"{OUT.parent.name}_val{round(100 * frac)}")
    d.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(d / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(part, fh, ensure_ascii=False, indent=2)
    print(f"{d.relative_to(ROOT)}: {len(rows)} questions | train {len(train)} | val {len(val)} "
          f"({len({r['subject'] for r in val})} subjects)")
    for name, part in (("train", train), ("val", val)):
        print(f"  {name} answers:", dict(sorted(collections.Counter(r["answer"] for r in part).items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction", type=float, help="also write <OUT dir>_val<N>/{train,val}.json")
    a = ap.parse_args()
    if a.val_fraction and OUT.exists():
        return write_split(a.val_fraction)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    source = []
    for split, f in FILES:
        rows = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)).to_pylist()
        print(f"{split}: {len(rows)} rows")
        source += rows
    rows, dropped = convert(source)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} questions | dropped {len(dropped)}: {dropped}")
    print("answers:", dict(sorted(collections.Counter(r["answer"] for r in rows).items())))
    if a.val_fraction:
        write_split(a.val_fraction)


if __name__ == "__main__":
    main()
