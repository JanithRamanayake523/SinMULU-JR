"""Validate the public data and build the labelled dev files from public train (Easy only).

  dev/archive/easy_full.jsonl   all 1,851 public-train questions
  dev/archive/easy_clean.jsonl  minus the questions with a near-duplicate in the test set (full Jaccard >= 0.5)
  dev/archive/inference_subset.jsonl  20 random questions per subject from easy_clean (14 x 20 = 280): screening set for
                              inference conditions (prompts, scoring, translation) before any training
  dev/archive/fewshot_pool.jsonl      5 fixed examples per subject (not in the subset), the only source of few-shot examples
  dev/archive/inference_rest.jsonl    the rest of easy_clean: confirmation set for the screening winners
Every dev row carries `overlaps_test`. The uid lists are also written to configs/subsets/ (tracked; dev/ is not).
Usage: python -m sinmmlu.build_dev [--per-subject 20 --shots 5 --seed 42]
"""
import argparse
import collections
import random

from . import dedup
from .data import ROOT, load_test, load_train, subject_si, write_jsonl

SUBSET_DIR = ROOT / "configs" / "subsets"

FIELDS = ("uid", "q_no", "subject", "category", "level", "file", "question", "choices")


def check(rows, name, labelled):
    bad = [r["uid"] for r in rows if len(r["choices"]) != 4
           or (labelled and int(r.get("answer", 0)) not in (1, 2, 3, 4))]
    keys = collections.Counter((r["level"], r["subject"], r["category"], r["q_no"]) for r in rows)
    dups = [k for k, v in keys.items() if v > 1]
    print(f"{name}: n={len(rows)} files={len({r['file'] for r in rows})} bad={len(bad)} duplicate_keys={len(dups)} "
          f"levels={dict(collections.Counter(r['level'] for r in rows))}")
    if labelled:
        print("  answers:", dict(sorted(collections.Counter(int(r["answer"]) for r in rows).items())))
    assert not bad and not dups, (bad[:5], dups[:5])


def slim(r):
    out = {k: r[k] for k in FIELDS}
    out.update(subject_si=subject_si(r), answer=int(r["answer"]), year=r.get("metadata", {}).get("year"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--shots", type=int, default=5, help="few-shot examples kept per subject")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train, test = load_train(), load_test()
    check(train, "public train", True)
    check(test, "public test", False)
    dup = dedup.overlapping_uids(train, test, remove_t=dedup.THRESHOLDS["public"][0])
    rows = [dict(slim(r), overlaps_test=r["uid"] in dup) for r in train]
    write_jsonl("dev/archive/easy_full.jsonl", rows)
    clean = [r for r in rows if not r["overlaps_test"]]
    write_jsonl("dev/archive/easy_clean.jsonl", clean)
    print(f"dev/archive/easy_full.jsonl n={len(rows)}; dev/archive/easy_clean.jsonl n={len(clean)} ({len(dup)} overlap the test set)")
    sub = inference_subset(clean, a.per_subject, a.seed)
    chosen = {r["uid"] for r in sub}
    pool = fewshot_pool([r for r in clean if r["uid"] not in chosen], a.shots, a.seed)
    chosen |= {r["uid"] for r in pool}
    rest = [r for r in clean if r["uid"] not in chosen]
    write_jsonl("dev/archive/inference_subset.jsonl", sub)
    write_jsonl("dev/archive/fewshot_pool.jsonl", pool)
    write_jsonl("dev/archive/inference_rest.jsonl", rest)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    for name, rs in (("inference_subset", sub), ("fewshot_pool", pool), ("inference_rest", rest)):
        (SUBSET_DIR / f"{name}_uids.txt").write_text("".join(r["uid"] + "\n" for r in rs), encoding="utf-8", newline="\n")
    print(f"dev/archive/inference_subset.jsonl n={len(sub)} ({a.per_subject} per subject, seed {a.seed}); "
          f"dev/archive/fewshot_pool.jsonl n={len(pool)} ({a.shots} per subject); dev/archive/inference_rest.jsonl n={len(rest)}; "
          f"uid lists -> {SUBSET_DIR.relative_to(ROOT)}")


def inference_subset(rows, per_subject, seed):
    """`per_subject` random questions from every subject (= source file), same seed -> same questions."""
    rng = random.Random(seed)
    by_file = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["file"], r["q_no"])):
        by_file[r["file"]].append(r)
    out = []
    for f in sorted(by_file):
        assert len(by_file[f]) >= per_subject, f"{f} has only {len(by_file[f])} questions"
        out += sorted(rng.sample(by_file[f], per_subject), key=lambda r: r["q_no"])
    return out


def fewshot_pool(rows, shots, seed):
    """`shots` examples per subject; the first four have four different answers (where possible) so that a k-shot
    prompt does not teach a position bias. Order inside a subject = the order used for k-shot (first k)."""
    rng = random.Random(seed + 1)
    by_file = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["file"], r["q_no"])):
        by_file[r["file"]].append(r)
    out = []
    for f in sorted(by_file):
        cands = by_file[f][:]
        rng.shuffle(cands)
        picked, seen = [], set()
        for r in cands:
            if r["answer"] not in seen:
                picked.append(r)
                seen.add(r["answer"])
        picked = picked[:min(4, shots)]
        picked += [r for r in cands if r not in picked][:shots - len(picked)]
        out += picked
    return out


if __name__ == "__main__":
    main()
