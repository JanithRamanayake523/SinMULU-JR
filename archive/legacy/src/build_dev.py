"""Build dev sets.
Team decision: public train is EVAL-ONLY, so ALL of it is Easy dev.
 - dev/dev_easy_full.jsonl   : all public train (1851). USE THIS for every model decision (CI ~ +-2 pp).
 - dev/dev_easy_screen.jsonl : ~150 subset stratified by source file, SMOKE TESTS ONLY (CI ~ +-7 pp).
 - --extra DIR: self-collected pre-2025 Medium/Hard JSON (same schema as the public files, with `answer`);
   builds dev/dev_medhard.jsonl (5-option items are reduced to 4 by dropping one random wrong option).
Usage: python src/build_dev.py [--extra DIR --per-level 120]"""
import argparse, collections, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dir, write_jsonl, drop_random_wrong_option, subject_si


def stratified(rows, n, rng):
    """Proportional sample, stratified by source file (one subject per file)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["file"]].append(r)
    total, out = len(rows), []
    for f, rs in sorted(by.items()):
        k = max(1, round(n * len(rs) / total))
        rng.shuffle(rs)
        out += rs[:k]
    return out


def slim(r):
    return {"q_no": r["q_no"], "subject": r["subject"], "subject_si": subject_si(r), "category": r["category"],
            "level": r["level"], "question": r["question"], "choices": r["choices"], "answer": int(r["answer"]),
            "year": r.get("metadata", {}).get("year"), "file": r["file"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra")
    ap.add_argument("--per-level", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    raw = load_dir("data/public train", level="easy")
    train = [slim(r) for r in raw]
    import dedup  # flag eval items that have a near-duplicate in the test set (full Jaccard >= 0.5); kept in dev, reported separately
    dup = {dedup.item_id(raw[x["cand"]]) for x in dedup.find_overlaps(raw, load_dir("data/public test")) if x["full"] >= 0.5}
    for r, rr in zip(train, raw):
        r["overlaps_test"] = dedup.item_id(rr) in dup
    print(f"overlaps_test flagged: {sum(r['overlaps_test'] for r in train)} -> dev_easy_clean n={sum(not r['overlaps_test'] for r in train)}")
    write_jsonl("dev/dev_easy_full.jsonl", train)
    scr = stratified(train, 150, rng)
    write_jsonl("dev/dev_easy_screen.jsonl", scr)
    print(f"easy full={len(train)} screen={len(scr)}")
    if a.extra:
        extra = [r for r in load_dir(a.extra) if r["level"] in ("medium", "hard")]
        out = []
        for lvl in ("medium", "hard"):
            rs = [slim(r) for r in extra if r["level"] == lvl]
            rs = [drop_random_wrong_option(r, rng) if len(r["choices"]) == 5 else r for r in rs]
            out += stratified(rs, a.per_level, rng)
        write_jsonl("dev/dev_medhard.jsonl", out)
        print("medium/hard dev:", len(out))
    else:
        print("No Medium/Hard dev yet (to be self-collected from pre-2025 papers; dedup against test first).")
