"""
Sinhala MMLU Shared Task - Final inference script
Team: SinBotics

The organizers will run this file exactly as it is, changing ONLY the DATA_REPO line
below to point to the final Sinhala MMLU dataset:

    pip install -r requirements.txt
    python inference.py

System: Qwen/Qwen3.5-4B + our LoRA adapter (rank 16, trained on Global-MMLU Sinhala + sinmulu_si), merged in bf16.
Each question gets the "cot_en_role_think" prompt: an expert Sri Lankan teacher of the subject, asked to think step
by step in English, with Qwen's thinking mode on, greedy decoding, at most 4,096 new tokens. The answer is then read
as the most probable of the digit tokens 1/2/3/4 after the reasoning + "\n\nAnswer: " (any stated answer is cut and
asked for again, so every question gets exactly one of 1-4). Deterministic: greedy decoding, fixed seed.

This is a self-contained copy of the code path that produced our final Codabench file
(training/sinmmlu: prompts.py cot_en_role_think, model.py LM.generate / label_logprobs_text, evaluate.py run_cot),
with the same batch sizes and length-sorted batching. Run time: about 5 h for 2,238 questions on one
NVIDIA A100 40 GB (generation batch 48 needs about 24 GB; on a CUDA out-of-memory a batch is split in halves).
"""

# ======================= ORGANIZERS CHANGE ONLY THIS LINE =======================
DATA_REPO = "SinhalaMMLU-SharedTask/Sinhala-MMLU-SharedTask-Test-Dataset"
# ================================================================================

OUTPUT_FILE = "predictions.jsonl"

# ----------------------- TEAM SETTINGS (keep exactly as used for your final submission)
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"     # released 2026-02-27, Apache-2.0
MODEL_ID = "SinhalaMMLU-SharedTask/team-sinbotics"     # this repo; our LoRA adapter is in its adapter/ folder
ADAPTER_SUBFOLDER = "adapter"
SEED = 42
GEN_BATCH = 48            # questions per generation batch (final run: 48 on an A100 40 GB)
SCORE_BATCH = 16          # questions per answer-scoring batch
MAX_NEW_TOKENS = 4096     # reasoning (thinking) length limit
FIXED_DATE = "26 Jul 2024"  # passed to the chat template so the prompt never depends on the run date

import json
import os
import random
import re
import unicodedata

import numpy as np
import torch
from huggingface_hub import HfFileSystem, list_repo_files


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============== 1. LOAD THE DATASET FROM HUGGING FACE (do not change) ===============
def load_questions(data_repo):
    fs = HfFileSystem()
    files = sorted(f for f in list_repo_files(data_repo, repo_type="dataset") if f.endswith(".json"))
    rows = []
    for f in files:
        with fs.open(f"datasets/{data_repo}/{f}", "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            raw = next(v for v in raw.values() if isinstance(v, list))
        for r in raw:
            r["_file"] = f
            rows.append(r)
    return rows


def get_level(r):
    return (r.get("metadata") or {}).get("difficulty") or r["_file"].split("/")[0]


# =============== 2. YOUR SYSTEM ===============
ZWNJ = "‌"
ANSWER_RE = re.compile(r"(?:පිළිතුර|Answer)\s*[:：]?\s*\**\s*([1-4])")
LABELS = ("1", "2", "3", "4")
CUE = "\n\nAnswer: "


def norm_model(text):
    """NFC, drop ZWNJ only (keep ZWJ, needed for conjuncts such as ප්‍ර), tidy spaces, keep newlines."""
    text = unicodedata.normalize("NFC", text).replace(ZWNJ, "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def subject_en(r):
    return " ".join(r["subject"].replace("_", " ").split())


def make_prompt(r):
    """cot_en_role_think (training/sinmmlu/prompts.py): teacher role + subject + English step-by-step instruction."""
    s = subject_en(r)
    opts = "\n".join(f"{l}. {norm_model(c)}" for l, c in zip(LABELS, r["choices"]))
    return (f"You are an expert Sri Lankan teacher of {s}.\n"
            f"The following is a multiple choice question in Sinhala about {s}.\n"
            "Think step by step in English, then choose the correct or most suitable answer from the options 1, 2, 3, 4.\n"
            "End your reply with \"Answer: <number>\".\n"
            f"\nQuestion: {norm_model(r['question'])}\n{opts}")


def load_system():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, MODEL_ID, subfolder=ADAPTER_SUBFOLDER).merge_and_unload()  # adds no parameters
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"model: {BASE_MODEL}@{BASE_REVISION[:8]} + {MODEL_ID}/{ADAPTER_SUBFOLDER} merged, {n:,} parameters, {dtype}", flush=True)
    return tok, model


def render(tok, r):
    msgs = [{"role": "user", "content": make_prompt(r)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE,
                                   enable_thinking=True)


def length_sorted_batches(ids_all, size):
    order = sorted(range(len(ids_all)), key=lambda i: len(ids_all[i]))
    return [order[b:b + size] for b in range(0, len(order), size)]


def oom_safe(fn, idx):
    """fn(idx) on one batch; on CUDA out-of-memory the batch is split in halves and retried."""
    try:
        fn(idx)
    except torch.cuda.OutOfMemoryError:
        if len(idx) == 1:
            raise
        torch.cuda.empty_cache()
        print(f"out of memory on a batch of {len(idx)}: splitting", flush=True)
        oom_safe(fn, idx[:len(idx) // 2])
        oom_safe(fn, idx[len(idx) // 2:])


def generate(tok, model, texts):
    """Greedy reasoning for every rendered prompt (length-sorted batches)."""
    ids_all = tok(texts, add_special_tokens=False)["input_ids"]
    out = [None] * len(ids_all)
    batches = length_sorted_batches(ids_all, GEN_BATCH)

    def run(idx):
        enc = tok.pad({"input_ids": [ids_all[j] for j in idx]}, padding=True, return_tensors="pt").to(model.device)
        g = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                           do_sample=False, temperature=None, top_p=None, top_k=None)
        for row, j in enumerate(idx):
            out[j] = tok.decode(g[row, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with torch.no_grad():
        for b, idx in enumerate(batches, 1):
            oom_safe(run, idx)
            print(f"generate: batch {b}/{len(batches)}", flush=True)
    return out


def label_token_ids(tok):
    out = []
    for d in LABELS:
        ids = tok.encode(d, add_special_tokens=False)
        assert len(ids) == 1, f"label {d!r} is not one token"
        out.append(ids[0])
    return out


def score_labels(tok, model, texts):
    """log P(label) of 1/2/3/4 as the next token after each text (left padding, positions from the mask)."""
    lab = label_token_ids(tok)
    ids_all = tok(texts, add_special_tokens=False)["input_ids"]
    out = [None] * len(ids_all)

    def run(idx):
        enc = tok.pad({"input_ids": [ids_all[j] for j in idx]}, padding=True, return_tensors="pt").to(model.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits")
        lp = torch.log_softmax(logits, dim=-1).cpu()
        for j, row in zip(idx, lp):
            out[j] = [row[v].item() for v in lab]

    with torch.no_grad():
        for idx in length_sorted_batches(ids_all, SCORE_BATCH):
            oom_safe(run, idx)
    return out


def predict_all(system, rows):
    tok, model = system
    texts = [render(tok, r) for r in rows]
    gens = generate(tok, model, texts)
    scored, n_stated = [], 0
    for t, g in zip(texts, gens):
        m = list(ANSWER_RE.finditer(g))
        n_stated += bool(m)
        reasoning = g[:m[-1].start()] if m else g  # cut the stated answer; the cue asks for it again
        if "</think>" not in reasoning:  # the tag is a special token and is decoded away (or thinking was cut off)
            reasoning = reasoning.rstrip() + "\n</think>"
        scored.append(t + reasoning.rstrip() + CUE)
    print(f"stated an answer: {n_stated}/{len(rows)}", flush=True)
    lps = score_labels(tok, model, scored)
    return [LABELS[int(np.argmax(lp))] for lp in lps]


# =============== 3. WRITE PREDICTIONS (do not change) ===============
if __name__ == "__main__":
    set_seed(SEED)
    rows = load_questions(DATA_REPO)
    print(len(rows), "questions loaded from", DATA_REPO)

    system = load_system()
    answers = predict_all(system, rows)

    assert len(answers) == len(rows), "predict_all() must return one answer per question"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r, a in zip(rows, answers):
            a = str(a).strip()
            assert a in {"1", "2", "3", "4"}, f"invalid answer {a!r} for q_no={r.get('q_no')}"
            f.write(json.dumps({
                "q_no": r["q_no"],
                "subject": r["subject"],
                "category": r["category"],
                "level": get_level(r),
                "answer": a,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("\nSaved", OUTPUT_FILE)
