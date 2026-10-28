"""Sinhala -> English translation of questions and options, cached per translator.

Translators (all open-weight, run inside our pipeline; their parameters count against the 8B limit):
  nllb    facebook/nllb-200-distilled-600M   0.6B, CC-BY-NC 4.0
  madlad  google/madlad400-3b-mt             ~3B, Apache 2.0
  self    the evaluated LLM itself (needs --model), e.g. backend name "self-gemma3_4b_it"
NLLB and MADLAD are sentence-level systems: text is split into lines and sentences, translated piece by piece and
re-joined. The LLM translates whole texts. Segments without Sinhala letters are copied unchanged.

  python -m sinmmlu.translate --backend nllb --data dev/archive/inference_subset.jsonl
  python -m sinmmlu.translate --backend self --model configs/models/gemma3_4b_it.yaml --data dev/archive/inference_subset.jsonl
Cache: cache/translations/<backend>.jsonl ({"src", "tgt"}) + <backend>.meta.json (model, revision, parameters).
Translations of test questions are derived test data: cache/ is gitignored.
"""
import argparse
import json
import re

from .data import ROOT, load_rows, norm_model

CACHE = ROOT / "cache" / "translations"
MT_MODELS = {"nllb": "facebook/nllb-200-distilled-600M", "madlad": "google/madlad400-3b-mt"}
SINHALA = re.compile(r"[඀-෿]")
SELF_PROMPT = ("Translate the following Sinhala text into English. Write only the English translation, nothing else.\n\n"
               "Sinhala text:\n{text}")


def backend_name(backend, model_cfg=None):
    return f"self-{model_cfg['name']}" if backend == "self" else backend


def texts_of(row):
    return [norm_model(row["question"])] + [norm_model(c) for c in row["choices"]]


def split_segments(text):
    """[[sentence, ...] per line]; sentences end with . ? ! (Sinhala uses the same marks)."""
    return [[s for s in re.split(r"(?<=[.?!])\s+", line) if s] for line in text.split("\n")]


def join_segments(lines):
    return "\n".join(" ".join(sents) for sents in lines)


# ----------------------------------------------------------------------------- cache
def load_cache(name):
    f = CACHE / f"{name}.jsonl"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        return {d["src"]: d["tgt"] for d in map(json.loads, fh)}


def append_cache(name, pairs):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{name}.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        for s, t in pairs:
            fh.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")


def attach(rows, name):
    """Rows with question_en / choices_en from the cache of translator `name`; raises if anything is missing."""
    cache = load_cache(name)
    missing = {t for r in rows for t in texts_of(r) if t not in cache}
    if missing:
        raise SystemExit(f"{len(missing)} texts not translated by {name!r}: run python -m sinmmlu.translate first")
    out = []
    for r in rows:
        q, *opts = [cache[t] for t in texts_of(r)]
        out.append(dict(r, question_en=q, choices_en=opts))
    return out


# ----------------------------------------------------------------------------- translators
class MT:
    """Sentence-level seq2seq translator (NLLB or MADLAD), beam search 4 (deterministic)."""

    def __init__(self, backend, batch_size=32):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from .model import count_params, resolve_dtype, resolve_revision
        self.backend, self.batch_size = backend, batch_size
        mid = MT_MODELS[backend]
        self.tok = AutoTokenizer.from_pretrained(mid, src_lang="sin_Sinh") if backend == "nllb" else AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(mid, dtype=resolve_dtype("auto")).to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()
        self.meta = {"model": mid, "revision_resolved": resolve_revision(mid, None, self.model),
                     "n_params_total": count_params(mid)}

    def _batch(self, batch):
        """Translate one batch; on CUDA out-of-memory (MADLAD + beam 4 on 24 GB) split it in halves and retry."""
        import torch
        src = batch if self.backend == "nllb" else ["<2en> " + s for s in batch]
        enc = self.tok(src, return_tensors="pt", padding=True).to(self.model.device)  # sentences: no truncation
        kw = dict(num_beams=4, max_new_tokens=256, max_length=None, do_sample=False)  # the checkpoints set max_length too
        if self.backend == "nllb":
            kw["forced_bos_token_id"] = self.tok.convert_tokens_to_ids("eng_Latn")
        try:
            with torch.no_grad():
                g = self.model.generate(**enc, **kw)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            del enc
            torch.cuda.empty_cache()
            print(f"[translate] out of memory on a batch of {len(batch)}: splitting", flush=True)
            half = len(batch) // 2
            return self._batch(batch[:half]) + self._batch(batch[half:])
        return [t.strip() for t in self.tok.batch_decode(g, skip_special_tokens=True)]

    def __call__(self, segments):
        from tqdm.auto import tqdm
        order = sorted(range(len(segments)), key=lambda i: len(segments[i]))  # similar lengths per batch
        out = [None] * len(segments)
        for b in tqdm(range(0, len(order), self.batch_size), desc=f"translate {self.backend} {len(segments)} segments",
                      unit="batch", dynamic_ncols=True, mininterval=2):
            idx = order[b:b + self.batch_size]
            for i, t in zip(idx, self._batch([segments[i] for i in idx])):
                out[i] = t
        return out


def translate_mt(texts, mt):
    pieces = sorted({s for t in texts for line in split_segments(t) for s in line if SINHALA.search(s)})
    done = dict(zip(pieces, mt(pieces))) if pieces else {}
    return [join_segments([[done.get(s, s) for s in line] for line in split_segments(t)]) for t in texts]


def translate_self(texts, lm, max_new_tokens=512):
    todo = [t for t in texts if SINHALA.search(t)]
    gen = dict(zip(todo, lm.generate([(("user", SELF_PROMPT.format(text=t)),) for t in todo], max_new_tokens)))
    return [gen.get(t, t) for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["nllb", "madlad", "self"])
    ap.add_argument("--data", nargs="+", required=True, help="dev .jsonl files and/or 'test'")
    ap.add_argument("--model", help="configs/models/<name>.yaml (backend self)")
    ap.add_argument("--limit", type=int, help="first N rows of each file only (smoke test)")
    a = ap.parse_args()
    mcfg = None
    if a.backend == "self":
        from .evaluate import load_model_cfg
        assert a.model, "--model is required for backend self"
        mcfg = load_model_cfg(a.model)
    name = backend_name(a.backend, mcfg)
    cache = load_cache(name)
    texts = sorted({t for d in a.data for r in load_rows(d)[:a.limit] for t in texts_of(r)} - set(cache))
    print(f"[translate] {name}: {len(cache)} cached, {len(texts)} to translate", flush=True)
    if not texts:
        return
    from .model import LM, env_info, load_lm, seed_everything
    seed_everything(42)
    if a.backend == "self":
        model, tok, info = load_lm(mcfg["model"], mcfg["revision"], mcfg["dtype"], mcfg["quant"])
        lm = LM(model, tok, chat=True, context=info["context_length"], gen_batch_size=mcfg.get("gen_batch_size", 16))
        meta = {"model": mcfg["model"], **info}
        tgt = translate_self(texts, lm)
    else:
        mt = MT(a.backend)
        meta = mt.meta
        tgt = translate_mt(texts, mt)
    append_cache(name, zip(texts, tgt))
    (CACHE / f"{name}.meta.json").write_text(json.dumps({**meta, "env": env_info()}, indent=2), encoding="utf-8")
    print(f"[translate] {name}: wrote {len(texts)} translations -> {CACHE / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
