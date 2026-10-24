"""Contamination check: candidate questions (train, generated, collected) against the test questions.

Similarity = character 5-gram Jaccard on norm_dedup() text, computed exactly for every pair found by either signal:
  full : question + options (options sorted, because years reorder them)
  q    : question only (only if >= MIN_Q_CHARS characters, to skip generic short stems)
Classes (thresholds by --source):
  public / generated : REMOVE full >= 0.50, REVIEW full 0.35-0.50
  collected          : REMOVE full >= 0.35, REVIEW full 0.25-0.35  (past papers recur, so stricter)
  all                : REVIEW also if q >= 0.5 and full >= 0.2; IGNORED if q >= 0.5 and full < 0.2 (generic stem)

Usage
  python -m sinmmlu.dedup                                   # public train vs public test
  python -m sinmmlu.dedup --candidates X.jsonl --source generated --report results/dedup_generated.txt
Writes <report> (uids + scores, no question text unless --show-text) and <report stem>_remove.jsonl.
"""
import argparse
import collections
import json

from .data import load_dir, load_jsonl, load_test, load_train, norm_dedup, path

N, MIN_Q_CHARS, QSIM = 5, 25, 0.5
THRESHOLDS = {"public": (0.5, 0.35), "generated": (0.5, 0.35), "collected": (0.35, 0.25)}  # (remove, review)


def grams(text):
    return {text[i:i + N] for i in range(len(text) - N + 1)} if len(text) >= N else {text}


def features(r):
    q = norm_dedup(r["question"])
    full = q + " " + " ".join(sorted(norm_dedup(c) for c in r["choices"]))
    return grams(full), (grams(q) if len(q) >= MIN_Q_CHARS else None)


def jaccard(a, b):
    if a is None or b is None:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def _hits(queries, refs, thr):
    """For each query set, the reference indices with Jaccard >= thr (inverted index; only used to FIND pairs)."""
    idx = collections.defaultdict(list)
    for j, s in enumerate(refs):
        for g in s or ():
            idx[g].append(j)
    found = []
    for s in queries:
        hits = set()
        if s:
            inter = collections.Counter(j for g in s for j in idx.get(g, ()))
            hits = {j for j, c in inter.items() if c / (len(s) + len(refs[j]) - c) >= thr}
        found.append(hits)
    return found


def find_overlaps(cands, test, find_thr=0.2):
    """All pairs with full >= find_thr or q >= QSIM, each with both exact Jaccards, most similar first."""
    cf, tf = [features(r) for r in cands], [features(r) for r in test]
    t_full, t_q = [f for f, _ in tf], [q for _, q in tf]
    by_full = _hits([f for f, _ in cf], t_full, find_thr)
    by_q = _hits([q for _, q in cf], t_q, QSIM)
    res = [{"cand": i, "test": j, "full": jaccard(cf[i][0], t_full[j]), "q": jaccard(cf[i][1], t_q[j])}
           for i in range(len(cands)) for j in by_full[i] | by_q[i]]
    return sorted(res, key=lambda x: -max(x["full"], x["q"]))


def classify(pair, remove_t, review_t):
    if pair["full"] >= remove_t:
        return "remove"
    if pair["full"] >= review_t or (pair["q"] >= QSIM and pair["full"] >= 0.2):
        return "review"
    if pair["q"] >= QSIM:
        return "ignored"
    return None


def overlapping_uids(cands, test, remove_t=0.5):
    return {cands[p["cand"]]["uid"] for p in find_overlaps(cands, test) if p["full"] >= remove_t}


def _load(spec, level=None):
    if spec == "train":
        return load_train()
    if spec == "test":
        return load_test()
    if str(spec).endswith(".jsonl"):
        rows = load_jsonl(spec)
        for i, r in enumerate(rows):
            r.setdefault("uid", f"{path(spec).name}/{i}")
        return rows
    return load_dir(spec, level=level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="train", help="'train', a dir of public-format *.json, or a .jsonl")
    ap.add_argument("--test", default="test", help="'test', a dir or a .jsonl")
    ap.add_argument("--source", choices=sorted(THRESHOLDS), default="public")
    ap.add_argument("--report", default="results/dedup_train_vs_test.txt")
    ap.add_argument("--show-text", action="store_true", help="include question text (do not publish such a report)")
    a = ap.parse_args()
    remove_t, review_t = THRESHOLDS[a.source]
    cands, test = _load(a.candidates), _load(a.test)
    groups = {"remove": [], "review": [], "ignored": []}
    for p in find_overlaps(cands, test):
        c = classify(p, remove_t, review_t)
        if c:
            groups[c].append(p)
    rm = sorted({cands[p["cand"]]["uid"] for p in groups["remove"]})
    lines = [f"Dedup: {a.candidates} (n={len(cands)}) vs {a.test} (n={len(test)}), source={a.source}",
             f"char {N}-gram Jaccard on norm_dedup(); full = question + sorted options; q = question only (>= {MIN_Q_CHARS} chars)",
             f"REMOVE (full >= {remove_t}): {len(groups['remove'])} pairs, {len(rm)} candidate items",
             f"REVIEW (full {review_t}-{remove_t}, or q >= {QSIM} and full >= 0.2): {len(groups['review'])} pairs",
             f"IGNORED generic stems (q >= {QSIM}, full < 0.2): {len(groups['ignored'])} pairs", ""]
    for title in ("remove", "review"):
        lines.append(f"== {title.upper()} ==")
        for p in groups[title]:
            c, t = cands[p["cand"]], test[p["test"]]
            lines.append(f"full {p['full']:.2f} q {p['q']:.2f}  {c['uid']}  <->  {t['uid']}")
            if a.show_text:
                lines.append(f"    cand: {c['question'].strip()[:120]}\n    test: {t['question'].strip()[:120]}")
        lines.append("")
    rep = path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    rep.with_name(rep.stem + "_remove.jsonl").write_text("".join(json.dumps({"uid": u}) + "\n" for u in rm), encoding="utf-8")
    print("\n".join(lines[:5]))
    print(f"-> {rep}")


if __name__ == "__main__":
    main()
