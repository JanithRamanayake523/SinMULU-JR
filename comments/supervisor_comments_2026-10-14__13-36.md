# Supervisor Comments — 2026-10-14 13:36

Re: `student_comment_2026-10-14_13-30.md`

Good work. The repo is clean, the validation is careful, and you were right to flag the subject mismatch and the Easy-only problem. I checked your code and claims against the data myself. Your counts are correct (1,851 / 2,238, 642 / 1,111 / 485, folder level = `metadata.difficulty` everywhere, 2,238 unique keys).

This note has four parts: **(A)** urgent findings and corrections, **(B)** answers to Q1–Q4, **(C)** code review, **(D)** next steps.

---

## A. Urgent — read first

### A1. Do NOT access the HF SinhalaMMLU dataset (correcting my §2.2)
I read the HF dataset card. It says that **for the ICTer shared task, the Hard test set is built from the paper's own 5-option Hard questions with one wrong option dropped.** So the HF data may contain the shared-task test questions, with answers. Using it, or even just requesting access, risks the *"use of test answers from any source"* clause. The rules also exclude *"people who have been given access to the full dataset."*

→ Your decision not to download it was **correct**, but my reason in §2.2 was wrong. **Change MANIFEST.md** to: `naist-nlp/SinhalaMMLU — NOT ACCESSED (may contain shared-task test items)`.

### A2. Train ↔ test overlap already exists
I ran a quick character-5-gram Jaccard check (question + choices) between public train and the test set. It found **8 near-duplicate pairs** (Jaccard 0.60–0.83). Sri Lankan exams reuse questions across years. Example:

| Train (2019) | Test (2025) |
|---|---|
| `Arts_humanities_easy_f_97.json` q78–82 | `Arts_Humanities_easy.json` q21–25 (5 consecutive items, the same block reused) |
| `Christianity_humanities_easy_f_130.json` q98, q105 | `Christianity_Humanities_easy.json` q53, q55 (options reversed) |
| `Geography_social_science_easy_f_117.json` q104 | `Geography_Social_science_easy.json` q15 |

Consequences:
1. A year cutoff alone (≤ 2024) does **not** guarantee clean data. Any past paper you collect will contain recurring questions, so **dedup is mandatory for every training example from any source**.
2. **My earlier threshold (Jaccard ≥ 0.8) is too strict.** It would have missed 7 of these 8. Use **≥ 0.5 = auto-remove, 0.35–0.5 = manual review**, with blank-runs and punctuation stripped first (see C, item 5).
3. If the team ever trains on public train, exclude these items. Tell the organisers about them through the team lead (point A4). It costs nothing and shows good faith.

### A3. Unicode shift between train and test
- **ZWNJ (U+200C):** 16% of test items (357/2,238) but only 1.6% of train (29/1,851). It's a legacy-font conversion artefact (e.g. `සැරසිල්ලක්‌`).
- **ZWJ (U+200D):** present in 76% of test items. It is **meaningful** in Sinhala (`ප්‍ර`, `ශ්‍රී`, `ක්‍ර`).

→ For model inputs, **strip ZWNJ, keep ZWJ.** Your `normalize()` strips both, which is fine for dedup but wrong for prompts (C, item 1).

### A4. Questions for the team lead to send the organisers (one email, today)
1. Is the Codabench hidden test exactly the 2,238 questions we received, or does it also include "SinhalaMMLU 2.0" Hard items (HF card)?
2. May we train on the released public train / dev set? The site says it's *"for building and validating systems."*
3. Must `subject` / `category` / `level` / `q_no` in `predictions.jsonl` match the test files verbatim, including inconsistent strings like `Eastern_Music` / `Eastern Music`?
4. We found 8 near-duplicates between public train and test (list attached). How should we handle them?
5. Is data generation with closed APIs allowed using models released after 1 Oct 2026, or does the date rule apply only to inference models?

Save the reply in `comments/organiser_reply_<date>.md`.

---

## B. Answers to your questions

### Q1. Subject name in the prompt
**Use the Sinhala `subject_original` field** (after `" ".join(s.split())`; several values have extra spaces, e.g. `'කෘෂි  විද්‍යාව'`, `'ගෘහ ආර්ථික  විද්‍යාව   '`). Reasons:
- The prompt is Sinhala, so a Sinhala subject name is the paper's setup and the natural form.
- The English names are inconsistent (`Eastern music` / `Eastern_music`, `Art` / `Arts`, `Health` vs `Health & Physical Education`), and some are wrong (`Health` has category `other` in Easy but `Social_science` in Medium).
- The Sinhala names exist for all 2,238 test items.

**Yes, run the ablation**, but only on one model and the **full** Easy dev: {no subject, English subject, Sinhala subject}. That's 3 runs of about 20 minutes each. Then freeze the prompt and stop tuning it.

### Q2. Compute plan for free Colab/Kaggle
Sizing for a T4 16 GB (prefer Kaggle: 2×T4 and a predictable 30 h/week quota):

| Job | Feasible? | Rough cost on one T4 |
|---|---|---|
| Log-prob eval, 7–8B 4-bit, 1,851 Qs | Yes | ~15–30 min (one forward pass per question) |
| Same with 4 cyclic permutations | Yes | ~1–2 h |
| Full test prediction (2,238) ×4 perms | Yes | ~1.5–2.5 h |
| QLoRA SFT, 7–8B, seq ≤ 512, ~5k examples, 1 epoch | Yes | ~2–4 h |
| QLoRA SFT, ~20k examples | Borderline | split across sessions, checkpoint every 500 steps |
| Continued pretraining on Sinhala text | **No** | drop it from the plan |

T4-specific rules:
- **No bf16 on T4.** Use `bnb_4bit_compute_dtype=torch.float16` and `fp16=True`. Watch for fp16 overflow (loss = NaN); if it happens, lower the LR. Also no FlashAttention-2, so use `attn_implementation="sdpa"`.
- P100: check that your bitsandbytes version supports 4-bit on Pascal before relying on it. Otherwise use T4 only.
- **Save checkpoints and results to persistent storage** (Kaggle output / Google Drive). Free sessions die without warning.
- Use a **3–4B model for fast iteration** on data-recipe experiments, then run only the final recipe on the 7–8B model.

### Q3. Contamination and subject strings
- **HF 2.0 overlap:** yes, ask (A4, Q1). Until the organisers answer, treat any pre-2025 **A/L** MCQ you collect as high-risk. Dedup it strictly, and record its source and year.
- **Subject strings:** copy **all four key fields (`q_no`, `subject`, `category`, `level`) verbatim from the test row**, with no normalisation. In the file I checked, two test files mix spellings (`Eastern_Music` / `Eastern Music`, `Health & Physical Education` / `Health and Physical Education`). Your keys are unique only *because* you keep them verbatim. Use the normalised Sinhala subject only *inside the prompt*.

### Q4. After baselines: how to build training data and validate

#### Q4a. Which datasets to create, by expected value

| # | Data | Value | Risk | Verdict |
|---|---|---|---|---|
| 1 | **Global-MMLU, Sinhala split** (`CohereLabs/Global-MMLU`, `si`; translated MMLU, Apache-2.0) | Ready today, ~14k 4-option Sinhala MCQs. Teaches the task format and general/STEM knowledge. | Translation is unnatural (paper §7, 71% naturalness). Content is US-centric. | **Use first.** It's your free "general" mix. |
| 2 | **Synthetic native Sinhala MCQs, per test subject × level**, generated by a strong closed LLM from the curriculum topic list (syllabus / textbook tables of contents, pre-2025) | Covers the **~30 test subjects** that have no data, at O/L and A/L style. This is the only route to Medium/Hard. | Hallucinated facts, especially culturally specific ones (Claude was only 39% on the cultural subset!). | **Main effort.** Run a pilot first (see Q4b). |
| 3 | **Self-collected pre-2025 past-paper MCQs with official answer keys** (e-Thaksalawa, provincial papers), grades 9–13 | Gold quality, native distribution. Also solves validation (Q4d). | Labour-intensive (OCR/manual). Exams recur, so strict dedup. A/L items risky until A4-Q1 is answered. | **Use mainly as dev data** (~100–150 per level), not as bulk training data. |
| 4 | Synthetic MCQs **derived from Sinhala text** (Wikipedia-si, textbooks): a closed LLM writes MCQs from a passage | Puts knowledge into the weights, a cheap substitute for continued pretraining (retrieval is banned, so training is the only way to add knowledge). | Same as #2, but grounded in the passage, so fewer hallucinations. | Second wave, if the pilot shows gains. |
| 5 | **Option-permutation augmentation** | Removes position bias, balances labels. | None. | Apply ×2 to all training data. |
| 6 | Format-only data | Near zero. Log-prob scoring doesn't need format learning, and SFT on #1 teaches the format anyway. | — | **Skip.** |
| 7 | Machine-translating more English MCQs yourselves | Global-MMLU already covers this, with better translation. | Worse terminology. | **Skip.** |

Also check **Belebele `sin_Sinh`** (900 Sinhala reading-comprehension MCQs) as a small, high-quality add-on. Also check whether **INCLUDE** has any Sinhala items, and if so, dedup them hard.

#### Q4b. Strategies to compare, in order (each step must beat the previous one on dev, or it gets dropped)

**Phase A — zero-shot, no training (15–17 Oct)**
1. 4–5 base models: raw prompt vs **chat template** (for Qwen3-style models set `enable_thinking=False`).
2. Prompt: the subject ablation (Q1). Strip ZWNJ vs not.
3. **Permutation averaging** (4 cyclic orders) vs single order.
4. **Calibration:** subtract the model's prior over 1/2/3/4 from a content-free input (e.g. question = "N/A"). This is cheap and fights position bias.
5. 3-shot vs 0-shot, one run only. The paper says it barely helps.

**Phase B — SFT (17–23 Oct), on a 3–4B model first, then repeat the best recipe on the chosen 7–8B model**

| Run | Training data | Loss target |
|---|---|---|
| B0 | none (best Phase-A setup) | — |
| B1 | Global-MMLU-si (~5k sample) + permutations | answer digit only |
| B2 | B1 + synthetic native **pilot** (~50 per subject-level) | answer digit only |
| B3 | B2 data | **answer digit first, then a short Sinhala explanation** (loss on both; inference still scores only the first digit) |
| B4 | best of B2/B3 with synthetic data scaled up (Q4e) | best target |

- B1 vs B0 tests whether SFT helps at all.
- B2 vs B1 tests whether native curriculum data helps beyond translated data.
- B3 vs B2 tests whether rationales help *without* adding generation cost at inference. Don't use chain-of-thought generation at inference: it's slow on a T4 and hard to reproduce.

**Phase C — freeze (24–26 Oct):** best SFT + permutation averaging + calibration if it helped.

#### Q4c. Data sources and rules
- **Generators:** any open model, or closed APIs (GPT / Claude / Gemini), but **disclose them in MANIFEST.md and the report.** For Sinhala, closed models are far better than ≤ 8B open models (paper: 67% vs 27%), so use a closed model for generation. Keep a log of `generator_model_id, date, prompt_template_version` per item.
- **Never show the generator any test question**, not even as a "style example." Style exemplars may come from public train *only if* you then dedup the generated items against public train too, so dev doesn't get contaminated. Better still, use your self-collected pre-2025 items.
- **Cutoff:** no source material dated 2025 or later. Store `source_year` per item.
- **Label quality pipeline:**
  1. Generator outputs question, 4 options, key, and a one-line justification.
  2. A **different** strong model answers blind, with the options shuffled. Keep the item only if it agrees with the key. Drop disagreements; don't try to repair them.
  3. Automatic filters: exactly one correct option; no "all of the above" / "none of the above"; ≥ 90% Sinhala-script characters; **correct option not systematically the longest** (a known generator artefact, so check its rate and keep it near 25%); balanced key positions after permutation.
  4. **Human spot-check:** native-speaker team members review 20 items per subject. If more than 10% are wrong, regenerate that subject with a better prompt or source material.
- **Dedup** (all generated + collected data, against both test and dev): normalise (C, item 5), then char-5-gram Jaccard on question + choices **≥ 0.5 = remove, 0.35–0.5 = manual check**. Also run question-only Jaccard, because options get reordered (see the Christianity example). Report the counts in MANIFEST.md and in the paper.

#### Q4d. How to validate Medium/Hard
Ranked:
1. **Best: a small gold dev set from pre-2025 official papers with official marking schemes**: ~120 Medium (O/L 2018–2024) + ~120 Hard (A/L 2018–2024), stratified over the test subjects, deduped against test. Two members can build it in about 2 days. At n = 120 the 95% CI is about ±8 pp, so it's only good for **large** effects (data-mix decisions, not prompt details). Build the **Medium set now**. Build the **Hard set only after** the organisers answer A4-Q1.
2. **Secondary: a synthetic held-out split** (10% of each generated subject-level, never trained on). It's biased toward the generator, but useful for spotting regressions.
3. Easy-only dev is **not acceptable** for decisions about Medium/Hard data.
4. **Never** use the test set for diagnostics (prediction distributions, confidence, anything). The leaderboard is for confirmation only, ≤ 1 submission/day, and the decision is written down from dev results *before* submitting.

#### Q4e. Size targets
SFT on a 7–8B model mostly **elicits** knowledge the model already has, and adds only a little new knowledge. Gains usually level off early, so *breadth of topics beats quantity per topic.*

| Data | Target | Why |
|---|---|---|
| Global-MMLU-si | 3–5k sampled (stratified) | more mostly adds US-centric content |
| Synthetic native, pilot | ~50 per test subject-level (~43 pairs → ~2k) | enough to measure the effect in B2 |
| Synthetic native, scaled (only if B2 > B1 by ≥ 2 pp on M/H dev) | 150–300 per subject-level (~6–12k), spread across syllabus topics | above this, returns diminish and the T4 budget runs out |
| After ×2 permutations | ≤ ~20k training examples, 1–2 epochs | fits 1–2 Kaggle sessions |

**API budget:** estimate the cost from the pilot (tokens per item × items) before scaling. Sinhala uses many tokens.

---

## C. Code review

| # | Severity | Location | Issue | Fix |
|---|---|---|---|---|
| 1 | **High** | [data_utils.py:18-21](src/data_utils.py:18) | `normalize()` removes **ZWJ** and its docstring says *"used for dedup and prompts."* Removing ZWJ breaks Sinhala conjuncts (`ප්‍ර`→`ප්ර`); your own prompt template has 3 ZWJs, and 76% of test items contain one. | Split into `norm_dedup()` (current behaviour + item 5) and `norm_model()` = NFC + **remove ZWNJ only** + collapse whitespace. Apply `norm_model` to question, choices *and* template consistently. |
| 2 | **High** (for the upcoming `eval_logprob.py`) | — | The most common bug in this setup is scoring the wrong digit token: `"1"` vs `" 1"` tokenise differently, and with a chat template the answer starts a new assistant turn. | `cands=[tok.encode(s,add_special_tokens=False) for s in ["1","2","3","4"]]` and `assert all(len(c)==1 ...)`. Check which variant (with/without a leading space) the model actually generates after your prompt (`generate(max_new_tokens=1)` on 20 items). Score `logits[0,-1, cand_ids]`. Run the sanity check: Qwen2.5-7B-Instruct should land at ~25–30% (the paper reports 27%). |
| 3 | Medium | [build_dev.py:31](src/build_dev.py:31) / plan | **Screen set n = 151 → 95% CI ≈ ±7 pp.** It can't tell models apart that differ by < 7 points. | Use the **full 1,851** for every model decision (CI ≈ ±2 pp, ~20 min/model). Keep the 151 set only for smoke tests. |
| 4 | Medium | [predict.py:14-18](src/predict.py:14) | The validator doesn't check `category`. Missing keys / bad JSON crash with a stack trace. A UTF-8 BOM (PowerShell's default output) breaks line 1. | Key on `(level, subject, category, q_no)`. Read with `encoding="utf-8-sig"`. Wrap the per-line parsing and report the line numbers. Check `type(q_no) is int`. |
| 5 | Medium (for `dedup.py`) | [data_utils.py:18](src/data_utils.py:18) | For dedup, blank-runs differ between years (85 dots in train vs 18 in test for the same Geography question), which lowers Jaccard. | `norm_dedup`: NFC, drop ZWJ/ZWNJ, **delete runs of `.` `_` `…` and all punctuation/digits-only option markers**, lowercase Latin, collapse whitespace. |
| 6 | Low | [build_dev.py:37](src/build_dev.py:37) | `slim(r) \| {"answer": r["answer"]}` overwrites `slim`'s `int()` cast with the raw value, so answer types can be mixed. This `--extra` branch was built for HF data, which is now off-limits. | Delete the override. Repurpose `--extra` for your self-collected pre-2025 Medium/Hard JSON (same schema). |
| 7 | Low | [build_dev.py:13](src/build_dev.py:13) | Stratifying by normalised English subject is fragile. | Stratify by source **file** (one subject per train file). |
| 8 | Low | all scripts | Relative paths (`"data/public train"`) only work from the repo root. | `ROOT = Path(__file__).resolve().parents[1]` and paths built from it. Organisers will run your code from somewhere else. |
| 9 | Low | [requirements.txt](requirements.txt) | Unpinned. | Pin **now**, to the versions on your Kaggle image (`pip freeze \| grep -E "torch\|transformers\|peft\|bitsandbytes\|accelerate"`). Library upgrades change logits. |
| 10 | Low | [configs/example_zeroshot.yaml](configs/example_zeroshot.yaml) | `dev: dev/dev.jsonl` doesn't exist; `revision: null`. | Point to `dev/dev_easy_full.jsonl`; pin the HF commit hash on first download. |
| 11 | Low | [MANIFEST.md](MANIFEST.md) | It says "awaiting organiser reply" for HF, which contradicts "no organiser email" and is now wrong (A1). | Update per A1, and add the 8 train–test overlaps. |

Correct as written: `permute_options` (label remap verified), `cyclic_perms`, `drop_random_wrong_option`, `load_dir` level handling, the `.gitignore` for test data, `write_jsonl` (no BOM, `\n`).

---

## D. About the "public train = eval only" decision

I respect the team lead's call, and it's the safe choice until the organisers answer. But be aware of the cost. The site describes the released set as *"for building and validating systems,"* and it's the **only native, gold-labelled Sinhala exam data you have.** If the organisers confirm it's allowed (A4-Q2), switch to **train on 70% / evaluate on the 30% that stays fixed** (stratified by file, excluding the 8 overlapping items). Native Easy data in the SFT mix is very likely worth several points on Easy. Until then, the plan in B works without it.

---

## E. Next steps

| Date | Task | Owner (suggested) |
|---|---|---|
| **14 Oct (today)** | Team lead sends the A4 email. Update MANIFEST (A1, overlaps). Pin requirements. | Lead / D |
| 14–15 Oct | Fix C1, C4, C6–C8. Write `dedup.py` with C5 normalisation + thresholds from A2. Run it on train vs test → `dedup_report.txt` (it should find my 8 pairs; if not, it has a bug). | A |
| 15–16 Oct | `eval_logprob.py` with the C2 checks. Sanity-check against the paper's Qwen2.5-7B-Instruct number on the **full** Easy dev. | B |
| 15–17 Oct | Phase A (B-Q4b). Pick the 3–4B iteration model and the 7–8B final model. | B |
| 15–17 Oct | Download Global-MMLU-si, Belebele-si. Dedup them. Build B1 data (×2 permutations, balanced keys). | C |
| 15–17 Oct | Collect the syllabus topic list per test subject-level (pre-2025). Write the generation + verifier prompts. Generate the **pilot** (~50 per subject-level) and run the Q4c quality pipeline. | A + D |
| 16–18 Oct | Build the **Medium gold dev** (~120, O/L ≤ 2024, deduped). Build the Hard one only after the organiser reply. | D |
| 17–19 Oct | `train_sft.py` (QLoRA, fp16, sdpa, checkpoint to persistent storage). Runs B1, B2. | C |
| **19 Oct** | First Codabench submission: best Phase-A zero-shot system (pipeline test). | D |
| 19–23 Oct | B3, B4 (scale synthetic data only if B2 helped on M/H dev). Final recipe on 7–8B. | C |
| 24 Oct | Start the report skeleton. | All |

## Bring to the next meeting
1. The organisers' reply, or confirmation that the A4 email was sent.
2. `dedup_report.txt` (train vs test).
3. Phase-A results table on the **full** Easy dev, including the Q1 subject ablation and the permutation averaging result.
4. 20 pilot synthetic items per subject (or the first 3 subjects) with spot-check error rates.
5. The tokeniser check output from C2 for each candidate model.

— Supervisor
