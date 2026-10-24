"""Analysis only (never used at inference): accuracy of every finished experiment by question type. Dev data only.

  python -m sinmmlu.qtypes --data dev/inference_subset.jsonl [--filter cot_en_role]
Types (heuristics on the question text; a question can have several):
  negation   a Sinhala negation / "wrong" word in the question (sinmmlu.negation.SI_NEGATION)
  suboption  labelled parts: at least two options name a part letter A-E or a Roman numeral (i-iv)
  blank      fill in the blank: "....", "____" or "…" in the question
  numeric    a digit in the question or the options
  plain      none of the above
"""
import argparse
import re

from .data import load_jsonl, norm_model
from .negation import SI_NEGATION
from .report import load_experiments

PART = re.compile(r"(?<![A-Za-z])[A-E](?![A-Za-z])|\b(?:i|ii|iii|iv)\b")
BLANK = re.compile(r"\.{3,}|_{3,}|…")
TYPES = ("negation", "suboption", "blank", "numeric", "plain")


def qtypes(r):
    q = norm_model(r["question"])
    t = set()
    if SI_NEGATION.search(q):
        t.add("negation")
    if sum(bool(PART.search(norm_model(c))) for c in r["choices"]) >= 2:
        t.add("suboption")
    if BLANK.search(q):
        t.add("blank")
    if re.search(r"\d", q + " ".join(r["choices"])):
        t.add("numeric")
    return t or {"plain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--filter", default="", help="only experiments whose id contains this text")
    a = ap.parse_args()
    rows = load_jsonl(a.data)
    types = {r["uid"]: qtypes(r) for r in rows}
    counts = {t: sum(t in v for v in types.values()) for t in TYPES}
    print(f"{a.data}: " + ", ".join(f"{t} {counts[t]}" for t in TYPES))
    exps = {e: v for e, v in load_experiments(a.data).items() if a.filter in e}
    print(f"{'experiment':<88} {'all':>6} " + " ".join(f"{t[:9]:>9}" for t in TYPES))
    for e, res in sorted(exps.items(), key=lambda x: -sum(x[1].values())):
        cells = []
        for t in TYPES:
            v = [ok for u, ok in res.items() if t in types[u]]
            cells.append(f"{sum(v) / len(v):9.3f}" if v else f"{'-':>9}")
        print(f"{e:<88} {sum(res.values()) / len(res):6.3f} " + " ".join(cells))


if __name__ == "__main__":
    main()
