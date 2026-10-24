# Student Comment — 2026-10-14 13:30

To: Supervisor. Re: `supervisor_comments_2026-10-14__12-22.md`, §1 and §2.1–2.2 (§2.3 dedup not started, as agreed).

## 1. Done

**§1 Repo setup** (nothing committed yet — awaiting your OK):
- Folders src configs results dev, plus `README.md`, `MANIFEST.md`, `requirements.txt` (unpinned for now), `.gitignore` (excludes `data/public test/`, zips, checkpoints), `results/results.csv` (your columns), `configs/example_zeroshot.yaml`.
- `src/data_utils.py`: loader, ZWJ/whitespace normaliser, Appendix-B prompt (subject / no-subject, adapted to "1, 2, 3, 4"), option permutation with label remap (tested), cyclic permutations, 5→4 option dropping (seeded).
- `src/validate_data.py`, `src/build_dev.py`, `src/predict.py` (format validator + zip: 2,238 unique keys, answers in "1".."4").
- `eval_logprob.py` and `train_sft.py` are stubs (§3, §4).

**§2 Dev set (adapted, see §2 and Q4):**
- Data check: train 1,851 Qs, 0 bad, answers 1:473 / 2:501 / 3:507 / 4:370. Test 2,238 (642/1,111/485), 0 bad, no duplicate keys.
- `dev/dev_easy_full.jsonl` (all 1,851 public train) and `dev/dev_easy_screen.jsonl` (151 Qs, subject-stratified, seed 42) for fast screening.
- `src/build_dev.py --extra <dir>` can build a Medium/Hard dev (150 each, 5→4 options) if such data ever becomes available; today it does not.

## 2. Settled facts and decisions (from the team lead, no answer needed)

1. **Data available:** only what is in `data/` (public train: Easy, with answers; public test: all levels, no answers). The original SinhalaMMLU on HF is gated and not available to us, so there is no download step and no labelled Medium/Hard data. §2.2 steps 1 and 3 are therefore not possible as written.
2. **Public train is evaluation only.** It is never used for training. I did not hold out 10%; all 1,851 questions are Easy dev. Any training data must be created separately by us, so §4.1 changes: there is currently no fine-tuning data, and synthetic data (§6.2) becomes the main data route.
3. **No organiser email.** If we need anything from the organisers, the team lead asks for us (see Q3).
4. **Compute:** free GPUs first (Colab / Kaggle), paid cloud possible later.

## 3. Observations that affect the plan

- **Easy-only dev vs the real metric.** Test is 29% Easy / 50% Medium / 22% Hard, and the metric is the macro average. With Easy-only dev, any model ranking is biased toward Easy, and we can't measure two thirds of the score.
- **Subject mismatch.** Train has 14 subjects; test has about 30. Only 576/642 Easy and 766/1,111 Medium test questions are in a subject that exists in train, and just 40/485 Hard. Physics, Chemistry, Biology, Economics, Business Studies, Political Science, Home Economics etc. have no train examples.
- **Inconsistent subject strings**, e.g. `Eastern music` / `Eastern_music`, `Health` / `Health & Physical Education` / `Health and Physical Education`, `Sinhala` vs `Sinhala language and literature`. For the submission file I plan to copy the test `subject` verbatim.

## 4. Questions for the supervisor

**Q1. Subject name in the prompt.** The paper's prompt puts the subject name in a Sinhala sentence, but our data gives English names. Should we normalise the English names, or use the Sinhala `subject_original` field? Do you want an early ablation of English vs Sinhala subject names?

**Q2. Compute plan.** With free Colab/Kaggle (T4/P100 16 GB, sessions of roughly 9–12 h), a 7–8B model in 4-bit fits for inference and QLoRA, but long runs and continued pretraining look unrealistic. Please size §3–§6 to that.

**Q3. Test contamination.** The HF card mentions a "2.0" Hard test set made from original Hard questions with one option removed. Is it worth asking the organisers (through the team lead) whether it overlaps our 2025 test? Also, must `subject` in `predictions.jsonl` match the test files verbatim?

**Q4. Main question: what to do after the base-model evaluation, and how to build training data.** We have no labelled Medium/Hard data and no training data. Please advise on:
  a. **What kinds of datasets we should create** (e.g. synthetic Sinhala MCQs per subject/level, translated English MCQs, MCQs derived from Sinhala textbooks/Wikipedia, option-permutation augmentation of our own created data, format-only data to teach answering with a digit).
  b. **Which strategies to compare** (e.g. zero-shot vs few-shot; SFT on synthetic data vs none; answer-digit-only loss vs full explanation; translated vs natively generated data; per-level data mix; continued pretraining on Sinhala text; permutation-averaged inference) and in which order, given ~13 days and free GPUs.
  c. **Data sources and rules:** which closed or open LLMs may generate data (disclosure required), the 2025 cutoff, how to check label quality (spot-checks, model agreement), and how to dedup generated data against the test set (§2.3).
  d. **How to validate Medium/Hard** without labels: hand-label a small set ourselves from pre-2025 papers, use synthetic held-out questions, or accept Easy-only dev?
  e. **Size targets:** how many examples per subject/level are worth creating before returns diminish for a 7–8B model.

## 4. Next
Awaiting your answers (especially Q4), then: start §2.3 `src/dedup.py` (train/dev vs test), and §3 baselines on the Easy dev.

— Student
