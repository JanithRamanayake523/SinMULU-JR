"""Data loading, prompt building, option permutation. Test answers never exist locally.
Paths are resolved from ROOT (repo root), so scripts work from any directory."""
import glob, json, os, random, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZWNJ, ZWJ = "‌", "‍"
LETTERS = ["1", "2", "3", "4"]

PROMPT_SUBJECT = (
    "පහත දැක්වෙන්නේ {subject} සම්බන්ධ බහුවරණ ප්‍රශ්නයකි.\n"
    "පහත දැක්වෙන ප්‍රශ්නය සඳහා නිවැරදි හෝ වඩාත්ම ගැළපෙන පිළිතුර 1, 2, 3, 4 පිළිතුරුවලින් තෝරන්න.\n\n"
    "ප්‍රශ්නය: {question}\n{options}\nපිළිතුර:"
)
PROMPT_NOSUBJECT = (
    "පහත දැක්වෙන්නේ බහුවරණ ප්‍රශ්නයකි.\n"
    "පහත දැක්වෙන ප්‍රශ්නය සඳහා නිවැරදි හෝ වඩාත්ම ගැළපෙන පිළිතුර 1, 2, 3, 4 පිළිතුරුවලින් තෝරන්න.\n\n"
    "ප්‍රශ්නය: {question}\n{options}\nපිළිතුර:"
)


def norm_model(text):
    """Model-input normalisation: NFC, remove ZWNJ ONLY (legacy-font artefact; ZWJ is meaningful in
    Sinhala conjuncts such as ප්‍ර), tidy spaces. Newlines are kept so sub-option lists stay readable."""
    text = unicodedata.normalize("NFC", text).replace(ZWNJ, "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def norm_dedup(text):
    """Dedup-only normalisation: NFC, drop ZWJ/ZWNJ, turn ALL punctuation/symbols into spaces (this also kills
    the blank-runs '....', '____', '…' whose length differs between years), lowercase Latin, collapse spaces."""
    text = unicodedata.normalize("NFC", text).replace(ZWJ, "").replace(ZWNJ, "")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def subject_si(row):
    """Sinhala subject name for the PROMPT only (stray spaces collapsed). Never used for prediction keys."""
    name = row.get("subject_si") or row.get("metadata", {}).get("subject_original") or row["subject"]
    return " ".join(norm_model(name).split())


def _abs(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_dir(path, level=None):
    """Load every *.json under `path`; adds `level` and `file` keys. `answer` may be absent (test)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(str(_abs(path)), "**", "*.json"), recursive=True)):
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                r = dict(r)
                r["level"] = level or r.get("metadata", {}).get("difficulty", "unknown")
                r["file"] = os.path.basename(f)
                rows.append(r)
    return rows


def load_jsonl(path):
    with open(_abs(path), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path, rows):
    path = _abs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_prompt(row, with_subject=True, subject_text=None):
    """Default subject = Sinhala subject_original (supervisor B-Q1); pass subject_text for the English ablation."""
    opts = "\n".join(f"{i + 1}. {norm_model(c)}" for i, c in enumerate(row["choices"]))
    tpl = PROMPT_SUBJECT if with_subject else PROMPT_NOSUBJECT
    return tpl.format(subject=subject_text or subject_si(row), question=norm_model(row["question"]), options=opts)


def permute_options(row, perm):
    """perm[i] = original index placed at new position i. Returns new row with remapped answer (if any)."""
    new = dict(row)
    new["choices"] = [row["choices"][j] for j in perm]
    if row.get("answer") is not None:
        new["answer"] = perm.index(int(row["answer"]) - 1) + 1
    return new


def cyclic_perms(n=4):
    return [[(i + s) % n for i in range(n)] for s in range(n)]


def random_perm(n, rng: random.Random):
    p = list(range(n)); rng.shuffle(p); return p


def drop_random_wrong_option(row, rng: random.Random):
    """5 -> 4 options by removing one random incorrect option (paper section 6)."""
    ans = int(row["answer"]) - 1
    drop = rng.choice([i for i in range(len(row["choices"])) if i != ans])
    keep = [i for i in range(len(row["choices"])) if i != drop]
    new = dict(row)
    new["choices"] = [row["choices"][i] for i in keep]
    new["answer"] = keep.index(ans) + 1
    return new
