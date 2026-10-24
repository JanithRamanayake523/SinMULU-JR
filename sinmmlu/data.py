"""Loading, text normalisation and option permutation.

Every row gets:
  level   easy | medium | hard (public train is all easy)
  file    source JSON file name; one file = one subject, so this is the subject key used for stratification
  uid     "<level>/<file>/q<q_no>", unique, used to join predictions back to rows
Test rows have no `answer`. Submission keys (q_no, subject, category, level) are always copied verbatim from the row,
because the subject strings are inconsistent inside the files (e.g. "Eastern_Music" and "Eastern Music").
"""
import glob
import json
import os
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "public train"
TEST_DIR = ROOT / "data" / "public test"
ZWNJ, ZWJ = "‌", "‍"
LEVELS = ("easy", "medium", "hard")


def path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------- normalisation
def norm_model(text):
    """Model input: NFC, drop ZWNJ only (a legacy-font artefact, far more common in test than in train), keep ZWJ
    (needed for conjuncts such as ප්‍ර), tidy spaces, keep newlines so sub-option lists stay readable."""
    text = unicodedata.normalize("NFC", text).replace(ZWNJ, "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def norm_dedup(text):
    """Dedup only: NFC, drop ZWJ/ZWNJ, punctuation and symbols to spaces, lowercase Latin, collapse spaces."""
    text = unicodedata.normalize("NFC", text).replace(ZWJ, "").replace(ZWNJ, "")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def subject_si(row):
    """Sinhala subject name for prompts (never for submission keys)."""
    name = row.get("subject_si") or row.get("metadata", {}).get("subject_original") or row["subject"]
    return " ".join(norm_model(name).split())


def subject_en(row):
    return " ".join(row["subject"].replace("_", " ").split())


def subject_family(row):
    """Coarse subject key shared by train and test spellings ("Arts"/"Art", "Health and Physical Science"/"Health",
    "Sinhala language and literature"/"Sinhala"): first three letters of the English subject name."""
    return re.sub(r"[^a-z]", "", row["subject"].lower())[:3]


# ----------------------------------------------------------------------------- io
def load_dir(d, level=None):
    """All *.json under `d` (recursive). `level` overrides metadata.difficulty (public train has none)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(str(path(d)), "**", "*.json"), recursive=True)):
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                r = dict(r)
                r["level"] = level or r.get("metadata", {}).get("difficulty", "unknown").lower()
                r["file"] = os.path.basename(f)
                r["uid"] = f"{r['level']}/{r['file']}/q{r['q_no']}"
                rows.append(r)
    uids = [r["uid"] for r in rows]
    assert len(set(uids)) == len(uids), f"duplicate (level, file, q_no) in {d}"
    return rows


def load_train():
    return load_dir(TRAIN_DIR, level="easy")


def load_test():
    return load_dir(TEST_DIR)


def load_jsonl(p):
    with open(path(p), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(p, rows):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_rows(spec):
    """'test' -> public test; anything else -> a dev .jsonl written by sinmmlu.build_dev."""
    return load_test() if spec == "test" else load_jsonl(spec)


def submission_key(r):
    return (r["level"], r["subject"], r["category"], r["q_no"])


# ----------------------------------------------------------------------------- options
def permute_options(row, perm):
    """perm[i] = original index shown at position i. The answer (if present) is remapped; translated options
    (`choices_en`) move with the originals."""
    new = dict(row)
    for k in ("choices", "choices_en"):
        if k in row:
            new[k] = [row[k][j] for j in perm]
    if row.get("answer") is not None:
        new["answer"] = perm.index(int(row["answer"]) - 1) + 1
    return new


def cyclic_perms(n=4):
    return [[(i + s) % n for i in range(n)] for s in range(n)]


def drop_random_wrong_option(row, rng: random.Random):
    """5 -> 4 options by removing one random wrong option (as in the paper, section 6)."""
    ans = int(row["answer"]) - 1
    drop = rng.choice([i for i in range(len(row["choices"])) if i != ans])
    keep = [i for i in range(len(row["choices"])) if i != drop]
    new = dict(row)
    new["choices"] = [row["choices"][i] for i in keep]
    new["answer"] = keep.index(ans) + 1
    return new
