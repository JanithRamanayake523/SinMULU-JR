# Supervisor Comments — 2026-10-14 12:22

**Status check:** Q64 fix verified (`Geography_social_science_easy_f_117.json`, q_no 64 → answer `3`). All 1,851 train labels are now valid (1: 473, 2: 501, 3: 507, 4: 370).

**Days left:** 13 until the leaderboard closes (27 Oct, 23:59). Every task below has a hard date. If something slips, tell me the same day.

---

## 0. Ground rules for the team (read first)

1. **Never look at test answers. Never tune on test questions. Never pick prompts using test questions.** This covers manually checking test predictions to "see what the model gets wrong." Test files are used only to produce `predictions.jsonl`.
2. **Year cutoff: no data from 2025 or later in training.** The whole test set comes from 2025 papers on sites like pastpapers.wiki. If you scrape anything, filter by year *and* deduplicate against the test set (see §2.3).
3. **Record everything you use** (model IDs with HF revision hashes, datasets, any closed-API use) in `MANIFEST.md` as you go. Writing it from memory on 28 Oct will be wrong.
4. **Reproducibility from day 1:** fix seeds, use greedy/log-prob decoding, pin library versions in `requirements.txt`. The organisers will re-run your code, and a run they can't reproduce is not ranked.
5. **The 8B budget covers the whole pipeline.** Base model plus LoRA plus anything else loaded at inference must total ≤ 8B.

---

## 1. Repo setup — **Today (14 Oct)**

Create this structure and commit it:

```
SinMULU-JR/
├── data/                  # (existing) do NOT commit test files publicly — test licence forbids redistribution
├── papers/
├── comments/
├── src/
│   ├── data_utils.py      # load JSON, build prompts, option permutation
│   ├── eval_logprob.py    # zero-shot / few-shot log-prob scoring
│   ├── train_sft.py       # LoRA/QLoRA fine-tuning
│   └── predict.py         # writes predictions.jsonl + zips it
├── configs/               # one YAML per experiment
├── results/               # one row per experiment in results.csv
├── MANIFEST.md
├── requirements.txt
└── README.md
```

- Add `data/public test/` to `.gitignore` if the repo will ever be public.
- `results/results.csv` columns: `exp_id, date, model, method, prompt, dev_easy, dev_medium, dev_hard, dev_macro, notes`.

---

## 2. Build a proper dev set — **14–15 Oct** (most important task this week)

### 2.1 The problem
Training data is **100% Easy**, but Easy is only 1/3 of the metric. A dev set built only from training data tells you nothing about Medium and Hard. Model choices made on it will be biased.

### 2.2 Action
1. **Download the original SinhalaMMLU** (`naist-nlp/SinhalaMMLU` on Hugging Face). It has about 7k questions across Easy, Medium and Hard, from before 2025.
2. **Email the organisers before using it** and ask: *"May we use the public SinhalaMMLU HF release (EMNLP 2025) for training/validation?"* Save the reply in `comments/`. Until they answer, use it **only as dev data**, not for training.
3. Build `dev.jsonl`: about 150 Easy, 150 Medium and 150 Hard, stratified by subject. **Hard questions in the original release have 5 options, but the test has 4.** Make a 4-option version by dropping one random wrong option (the paper's §6 did the same), and keep a fixed seed.
4. Hold out **10% of the public train** (stratified by subject) as a second Easy dev set.

### 2.3 Contamination check (mandatory)
Write `src/dedup.py` to compare every training/dev question against all test questions:
- exact match after normalising whitespace and ZWJ (`‍`)
- character 5-gram Jaccard ≥ 0.8, **or** embedding cosine ≥ 0.95 (the paper's threshold)

Remove any matches from training. Report the count in `MANIFEST.md` and later in the paper. This protects you from disqualification.

**Deliverable:** `dev.jsonl`, `dedup_report.txt`, organiser email sent.

---

## 3. Baselines — **15–17 Oct**

Write `eval_logprob.py` to score the option tokens (`1`/`2`/`3`/`4`) after `පිළිතුර:` and take the argmax. Use the paper's Appendix B Sinhala prompt **with the subject name** (the paper shows the subject name helps).

Run zero-shot on the dev set, with **4-bit only for screening**:

| Priority | Candidates (all ≤ 8B, released before 1 Oct 2026) |
|---|---|
| Must | Qwen2.5-7B-Instruct (paper baseline ≈ 27%, use it as the sanity check) |
| Must | Latest Qwen family ≤ 8B instruct model |
| Must | Gemma family ≤ 8B (strong multilingual tokeniser) |
| Should | Llama-3.1-8B-Instruct, Aya Expanse 8B (paper baselines) |
| Could | Any Sinhala- or Indic-adapted open model ≤ 8B you can find on HF (check release date and licence) |

For each model also report:
- **Tokeniser fertility:** average tokens per Sinhala word on 1,000 training questions. Lower is better, and it predicts how much the model has actually learned Sinhala.
- **Answer-position bias:** a histogram of predicted options. If one option gets more than 35% of predictions, note it.

**Sanity check:** if Qwen2.5-7B-Instruct isn't roughly 25–30% on dev, the prompt or scoring code has a bug. Fix that before anything else.

**Deliverable:** results table in `results.csv`. **Pick the top 2 base models by 17 Oct.** Don't run more than two base models through fine-tuning, because there isn't time.

---

## 4. First fine-tuning run — **17–21 Oct**

### 4.1 Data
- Public train (minus the 10% held out), plus original SinhalaMMLU **if the organisers approve** (minus the dev split, deduplicated).
- **Option-permutation augmentation:** shuffle the options 2–3× per question and remap the gold label. This multiplies the data and removes position bias. Balance the gold labels evenly across 1/2/3/4 (the current train has too few 4s).

### 4.2 Recipe (starting point, don't over-tune)
- QLoRA, rank 16–64, α = 2×rank, all linear layers, LR 1e-4 to 2e-4, cosine schedule, 2–3 epochs, max length 512.
- Train the target as **only the answer digit** (loss on the answer token only). That matches log-prob scoring at inference.
- Remember that LoRA parameters count toward 8B. A 7B base plus a rank-64 adapter is fine, but check the arithmetic in `MANIFEST.md`.

### 4.3 What to check
- Dev Easy *and* Medium/Hard, separately. Easy-only training data may raise Easy and leave Hard flat or lower. If so, that's a finding to report in the paper.
- Training loss vs dev accuracy each epoch, to catch overfitting on a small dataset.

**Deliverable:** SFT checkpoint, results row, a short note in `comments/` on what changed.

---

## 5. First Codabench submission — **by 19 Oct**

Submit the best **zero-shot baseline** before fine-tuning finishes. Reasons:
- It tests the full `predictions.jsonl` → `predictions.zip` pipeline early. The format is `{"q_no":1,"subject":"Arts","category":"humanities","level":"easy","answer":"2"}`, and `answer` is a **string**.
- It gives you a reference point on the real test distribution.

Validate the file before uploading: 2,238 lines (642 + 1,111 + 485), unique `(level, subject, q_no)` keys, every answer in `{"1","2","3","4"}`.

**Submission budget:** 3 per day. Plan at most **1 submission per day**, chosen from dev results, never by trying things on the leaderboard.

---

## 6. Improvements, in priority order — **21–25 Oct**

Do these in order, and stop when time runs out:

1. **Permutation-averaged inference:** score each question under 4 cyclic option orders and average the probabilities, mapping back to the original options. Zero training cost, and it usually adds a few points while removing position bias.
2. **Medium/Hard data:** if the organisers allowed the original SinhalaMMLU, check that its Medium/Hard parts are in training. Otherwise, the next best source is **synthetic O/L and A/L-style MCQs** generated from **pre-2025** Sinhala textbooks and curriculum material. If a closed API generates them, disclose it in the manifest and report. Spot-check 50 by hand for correctness, since wrong synthetic labels hurt.
3. **Continued pretraining on Sinhala text**, only if there's time and GPU: a short LoRA pass on clean Sinhala text (Wikipedia, CulturaX/MADLAD si) before SFT. It targets the cultural-knowledge gap the paper identified. It's expensive, so skip it if compute is limited.
4. **Targeted error analysis on dev** (never test): split dev accuracy by negation questions (නොගැළපේ, අසත්‍ය, වැරදි, සාවද්‍ය) and by sub-option questions, as the paper did in Table 5. If either is much worse, add augmented examples of that type.

**Don't do:** retrieval of any form, ensembles that exceed 8B, CoT generation that makes inference slow and non-deterministic (unless dev shows a big gain and you fix the seed), or any manual step at inference.

---

## 7. Freeze — **26–27 Oct**

- **26 Oct:** pick the final system *by dev macro score*, not by the leaderboard. Make the final submission.
- **27 Oct:** submit a backup only if needed. **The deadline is 23:59.** Submit by noon, not at 23:50.
- Tag the commit: `git tag final-submission`.

---

## 8. Code package and report — **28–30 Oct**

- **28 Oct, code package:** a fresh-machine test. Someone who didn't write the code clones the repo, runs `pip install -r requirements.txt && python src/predict.py --config configs/final.yaml`, and gets *identical* predictions. Include the LoRA weights or a download link, HF revision hashes, and the hardware used.
- **`MANIFEST.md`:** every model with its parameter count (show the sum ≤ 8B), every dataset, the dedup result, and closed-API disclosure (or "none").
- **30 Oct, system description report and slides.** Suggested structure:
  1. Introduction (task, 8B constraint)
  2. Data (the Easy-only train gap, how you filled it, dedup)
  3. System (base model choice backed by fertility and zero-shot results, SFT recipe, permutation inference)
  4. Results (dev + test, per-level table, ablations: +SFT, +permutation, +extra data)
  5. Analysis (negation, sub-options, per-subject, cultural vs general)
  6. Limitations
  - **You must cite Pramodya et al., EMNLP 2025** (BibTeX is on the Task page) in both the report and the slides.

Start the report skeleton on **24 Oct**, while the final experiments run. Don't leave it all to the last two days.

---

## Suggested role split (adjust to team size)

| Member | Owns |
|---|---|
| A | Data: dev set, dedup, augmentation, organiser email, MANIFEST |
| B | Eval: `eval_logprob.py`, baselines, results table, error analysis |
| C | Training: SFT pipeline, checkpoints, permutation inference |
| D | Submission and report: `predict.py`, format validation, Codabench, report/slides |

---

## Next meeting — bring these
1. The results table with ≥ 4 zero-shot baselines on the new dev set (split by level).
2. Tokeniser fertility numbers.
3. The dedup report.
4. The organisers' reply about the original SinhalaMMLU, or confirmation the email was sent.
5. Your available compute (GPU type, VRAM, hours per day). I'll adjust §4 and §6 to fit it.

— Supervisor
