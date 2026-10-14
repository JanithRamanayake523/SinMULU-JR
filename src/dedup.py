"""Contamination check: compare candidate (train / dev / generated / collected) questions against test questions.

Similarity = character 5-gram Jaccard on norm_dedup() text, computed two ways for EVERY candidate pair found by either signal
(both values are always exact, never a thresholded 0):
  full : question + options (options sorted, because years reorder them)
  q    : question only (needs >= MIN_Q_CHARS chars, to avoid generic short stems)

Rules by --source (supervisor 16:35, C-Q5):
  public/generated : REMOVE full >= 0.5 ; REVIEW full 0.35-0.5
  collected        : REMOVE full >= 0.35; REVIEW full 0.25-0.35   (past papers recur, so stricter)
  all sources      : REVIEW also if q >= 0.5 and full >= 0.2 (same stem, options partly changed)
                     IGNORED if q >= 0.5 and full < 0.2 (generic stem such as "choose the correctly spelled word")

Usage:
  python src/dedup.py                                      # public train vs public test -> dedup_report.txt
  python src/dedup.py --candidates DIR_OR_JSONL --source collected --test dev/dev_easy_full.jsonl --report dedup_x.txt
  python src/dedup.py --show-text                          # adds text (report then contains test text: do not publish)
Writes <report> (ids + scores) and <report stem>_remove.jsonl (ids of candidates to drop)."""
import argparse, collections, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import ROOT, load_dir, load_jsonl, norm_dedup

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


def item_id(r):
    return f"{r['file']}:q{r['q_no']}"


def build_index(sets):
    idx = collections.defaultdict(list)
    for i, s in enumerate(sets):
        if s:
            for g in s:
                idx[g].append(i)
    return idx


def candidates_from(sets, idx, ref_sets, thr):
    """Reference indices whose Jaccard with each query set is >= thr (used only to FIND pairs)."""
    found = []
    for s in sets:
        hits = set()
        if s:
            inter = collections.Counter(j for g in s for j in idx.get(g, ()))
            hits = {j for j, c in inter.items() if c / (len(s) + len(ref_sets[j]) - c) >= thr}
        found.append(hits)
    return found


def find_overlaps(cands, test, find_thr=0.2):
    """All pairs with full >= find_thr or q >= QSIM, each with BOTH exact Jaccards."""
    cf = [features(r) for r in cands]
    tf = [features(r) for r in test]
    t_full, t_q = [f for f, _ in tf], [q for _, q in tf]
    by_full = candidates_from([f for f, _ in cf], build_index(t_full), t_full, find_thr)
    by_q = candidates_from([q for _, q in cf], build_index(t_q), t_q, QSIM)
    res = []
    for i in range(len(cands)):
        for j in by_full[i] | by_q[i]:
            full, q = jaccard(cf[i][0], t_full[j]), jaccard(cf[i][1], t_q[j])
            res.append({"cand": i, "test": j, "full": full, "q": q})
    return sorted(res, key=lambda x: -max(x["full"], x["q"]))


def classify(r, remove_t, review_t):
    if r["full"] >= remove_t:
        return "remove"
    if r["full"] >= review_t or (r["q"] >= QSIM and r["full"] >= 0.2):
        return "review"
    if r["q"] >= QSIM:
        return "ignored"
    return None


def load_items(path, level=None):
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if str(path).endswith(".jsonl"):
        rows = load_jsonl(path)
        for r in rows:
            r.setdefault("file", p.name)
        return rows
    return load_dir(path, level=level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/public train", help="dir of public-format *.json, or a .jsonl")
    ap.add_argument("--test", default="data/public test", help="dir or .jsonl to compare against")
    ap.add_argument("--source", choices=sorted(THRESHOLDS), default="public")
    ap.add_argument("--report", default="dedup_report.txt")
    ap.add_argument("--show-text", action="store_true")
    a = ap.parse_args()
    remove_t, review_t = THRESHOLDS[a.source]
    test = load_items(a.test)
    cands = load_items(a.candidates, level="easy" if a.candidates == "data/public train" else None)
    pairs = find_overlaps(cands, test)
    groups = {"remove": [], "review": [], "ignored": []}
    for r in pairs:
        c = classify(r, remove_t, review_t)
        if c:
            groups[c].append(r)
    rm_ids = sorted({item_id(cands[r["cand"]]) for r in groups["remove"]})

    L = [f"Dedup report: candidates={a.candidates} (n={len(cands)}) vs {a.test} (n={len(test)}), source={a.source}",
         f"Method: char {N}-gram Jaccard on norm_dedup(); full=question+sorted options, q=question only (>= {MIN_Q_CHARS} chars); both exact for every pair",
         f"REMOVE (full >= {remove_t}): {len(groups['remove'])} pairs, {len(rm_ids)} distinct candidate items",
         f"REVIEW (full {review_t}-{remove_t}, or q >= {QSIM} with full >= 0.2): {len(groups['review'])} pairs",
         f"IGNORED generic stems (q >= {QSIM}, full < 0.2): {len(groups['ignored'])} pairs", ""]
    for title in ("remove", "review"):
        L.append(f"== {title.upper()} ==")
        for r in groups[title]:
            c, t = cands[r["cand"]], test[r["test"]]
            L.append(f"full {r['full']:.2f} q {r['q']:.2f}  cand {item_id(c)} [{c.get('metadata', {}).get('year', c.get('year', '?'))}]"
                     f"  <->  test {t.get('level', '?')}/{item_id(t)}")
            if a.show_text:
                L.append(f"    cand: {c['question'].strip()[:120]}\n    test: {t['question'].strip()[:120]}")
        L.append("")
    rep = ROOT / a.report
    rep.write_text("\n".join(L), encoding="utf-8", newline="\n")
    rem = rep.with_name(rep.stem + "_remove.jsonl")
    rem.write_text("".join(json.dumps({"id": i}) + "\n" for i in rm_ids), encoding="utf-8")
    print("\n".join(L[:5]))
    print(f"-> {rep.name}, {rem.name}")


if __name__ == "__main__":
    main()
