"""Analysis only (never used at inference): negation questions and translation quality. Dev data only.

  python -m sinmmlu.negation --data dev/archive/inference_subset.jsonl [--samples 20]
1. Accuracy of every finished experiment on questions with / without a Sinhala negation word (the paper names
   negation as a main difficulty).
2. For each cached translator: of the questions with a negation word, how many English translations still contain
   an English negation ("not", "except", "incorrect", ...).
3. A side-by-side file with N random questions and their translations (outputs/analysis/translations_<data>.md,
   gitignored because it contains benchmark text).
"""
import argparse
import random
import re

from .data import ROOT, load_jsonl, norm_model, path
from .report import load_experiments
from .translate import CACHE, load_cache

# Sinhala negation / "which is wrong" markers in question stems (analysis heuristic, not exhaustive)
SI_NEGATION = re.compile(r"නොවන|නොවේ|නොගැළපේ|නොගැලපේ|නොගැළපෙන|නොගැලපෙන|නොමැති|නොකරන|නොවූ|වැරදි|අසත්‍ය|සාවද්‍ය")
EN_NEGATION = re.compile(r"\b(not|no|never|except|incorrect|false|wrong|untrue|least|cannot|neither|nor)\b|n't",
                         re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--samples", type=int, default=20)
    a = ap.parse_args()
    rows = load_jsonl(a.data)
    neg = {r["uid"] for r in rows if SI_NEGATION.search(norm_model(r["question"]))}
    print(f"{a.data}: {len(neg)} of {len(rows)} questions contain a Sinhala negation word")

    print("\n1. accuracy with / without negation")
    print(f"{'experiment':<95} {'all':>6} {'neg':>6} {'no neg':>7}")
    exps = load_experiments(a.data)
    for e, res in sorted(exps.items(), key=lambda x: -sum(x[1].values())):
        n_ = [v for u, v in res.items() if u in neg]
        p_ = [v for u, v in res.items() if u not in neg]
        print(f"{e:<95} {sum(res.values()) / len(res):6.3f} {sum(n_) / max(len(n_), 1):6.3f} {sum(p_) / max(len(p_), 1):7.3f}")

    print("\n2. negation kept by the translation (questions with a Sinhala negation word)")
    caches = {f.stem: load_cache(f.stem) for f in sorted(CACHE.glob("*.jsonl"))}
    for name, cache in caches.items():
        qs = [norm_model(r["question"]) for r in rows if r["uid"] in neg]
        have = [q for q in qs if q in cache]
        kept = sum(bool(EN_NEGATION.search(cache[q])) for q in have)
        print(f"{name:<22} translated {len(have)}/{len(qs)}; English negation present in {kept}/{max(len(have), 1)}"
              f" ({kept / max(len(have), 1):.0%})")

    out = ROOT / "outputs" / "analysis" / f"translations_{path(a.data).stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    pick = rng.sample(rows, min(a.samples, len(rows)))
    lines = [f"# Translations of {a.samples} random questions from {a.data} (dev data; do not publish)\n"]
    for r in pick:
        q = norm_model(r["question"])
        lines.append(f"## {r['uid']}  (gold {r['answer']}{', negation' if r['uid'] in neg else ''})\n")
        lines.append(f"**Sinhala:** {q}\n")
        for i, c in enumerate(r["choices"], 1):
            lines.append(f"{i}. {norm_model(c)}")
        for name, cache in caches.items():
            lines.append(f"\n**{name}:** {cache.get(q, '(not translated)')}")
            for i, c in enumerate(r["choices"], 1):
                lines.append(f"{i}. {cache.get(norm_model(c), '(not translated)')}")
        lines.append("\n---\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n3. side-by-side translations -> {out}")


if __name__ == "__main__":
    main()
