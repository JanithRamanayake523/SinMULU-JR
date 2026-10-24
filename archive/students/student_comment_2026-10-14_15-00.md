# Student Comment — 2026-10-14 15:00

To: Supervisor. Re: `supervisor_comments_2026-10-14__13-36.md`.

## 1. Done (nothing committed yet)

| Item | Status |
|---|---|
| A1 | MANIFEST says HF SinhalaMMLU = **NOT ACCESSED** (may contain test items). We never requested it. |
| A2 | `src/dedup.py` written and run on train vs test → `dedup_report.txt`. See §2. |
| A3 | `norm_model()` (NFC, remove **ZWNJ only**, collapse spaces) for prompts; `norm_dedup()` for dedup. |
| C1 | Prompt now uses `norm_model` on question, choices and subject; ZWJ verified kept (tested). |
| C4 | `predict.py`: keys `(level, subject, category, q_no)` verbatim, `utf-8-sig`, per-line error report, `q_no` int check. |
| C5 | `norm_dedup`: all punctuation/symbols → space (kills blank-runs), ZWJ/ZWNJ dropped, Latin lowercased. |
| C6 | Removed the `answer` override in `build_dev.py`; `--extra` now means *self-collected pre-2025 Medium/Hard JSON*. |
| C7 | Dev screen set stratified by source **file**. |
| C8 | All paths resolved from `ROOT` (`data_utils.py`), so scripts run from any directory. |
| C10, C11 | Config points to `dev/dev_easy_full.jsonl`; MANIFEST updated with A1 and the dedup counts. |
| B-Q1 | Default prompt subject = Sinhala `subject_original` (spaces collapsed). `dev_*.jsonl` carry a `subject_si` field. English subject is only an ablation option. |
| C3 | Screen set kept for smoke tests only; decisions use the full 1,851. |

Not done yet: **C9 (pin requirements)** — I don't know the Kaggle image versions; needs `pip freeze` from a Kaggle notebook. C2 goes into `eval_logprob.py` (next).

## 2. Dedup result (train vs test)
- **REMOVE (full ≥ 0.5): 18 pairs / 18 train items.** All your examples are in it (Arts q78–82, Christianity q98/q105, Geography q104), plus 10 more (Catholicism ×3, Civics ×2, Geography q26, Health q52, Drama ×2, Dancing q41).
- **REVIEW: 41 pairs.** **IGNORED: 80 pairs.**
- **Why "ignored":** your suggestion to also use question-only Jaccard at ≥ 0.5 flagged 120 pairs. Most are generic stems with different options, e.g. "නිවැරදි අක්ෂර වින්‍යාසයෙන් යුතු වචනය තෝරන්න" (choose the correctly spelled word) appears in many items of train and test with full-text Jaccard 0.00. These aren't duplicates. So the rule is: REMOVE = full ≥ 0.5; REVIEW = full 0.35–0.5, or question-only ≥ 0.5 with full ≥ 0.2; question-only matches with full < 0.2 are counted, not listed.
- The report holds only ids and scores (no test text), so it can stay in the repo. `--show-text` adds text for manual review.
- It runs in ~5 s on 1,851 × 2,238 items.

## 3. Questions

**Q1. Dedup rule.** Are you OK with ignoring question-only matches when the options differ (full < 0.2)? Otherwise the generic Sinhala-grammar/instruction stems would remove a large share of any generated or collected data.

**Q2. The 41 REVIEW pairs and the threshold for generated data.** Synthetic items from a closed LLM can paraphrase a test question without sharing 5-grams. Is char 5-gram Jaccard enough for generated data, or should we add the embedding check (cosine ≥ 0.95) you mentioned earlier? That needs an embedding model: which one is allowed and cheap on Kaggle (multilingual-e5-small / LaBSE)?

**Q3. Keeping the 18 overlapping items in Easy dev.** Since public train is eval-only, I keep all 1,851 in dev. Those 18 have near-twins in test, so they measure memory of a test item as much as ability if we later train on similar sources. I plan to report Easy dev both with and without them. OK, or should I drop them from dev now?

**Q4. `norm_model` keeps newlines.** I collapse only runs of spaces/tabs and keep line breaks, so sub-option questions (A, B, C lines) stay readable. You said "collapse whitespace"; is keeping newlines OK?

**Q5. Overlap and the leaderboard.** If ≥ 18 Easy test items (~3% of Easy) repeat earlier exam questions, a model that has seen past papers (e.g. through our Phase-B data) gets a small advantage. Any past paper we collect is therefore the same risk. Do you want a stricter dedup threshold for collected data than for generated data?

**Q6. Gold dev workload.** The Medium gold dev needs ~120 O/L items 2018–2024 with official keys, stratified over test subjects. Which subjects first? Test Medium has ~30 subjects; 120 items is ~4 per subject. Should we weight by test size, or cover a smaller set of the biggest subjects?

**Q7. For the organisers (through the team lead, whenever it is sent).** Add one item to A4: are open third-party datasets (Global-MMLU-si, Belebele) allowed as training data? You treat them as fine; I want it in writing before Phase B.

## 4. Next (per your plan E)
`eval_logprob.py` with the C2 token checks, then Phase A on the full Easy dev, including the subject-name ablation. Data generation and any dataset downloads are postponed by the team lead until the next step.

— Student
