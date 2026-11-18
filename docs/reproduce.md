# Reproducing the reported results

Each result in the report (`report/`) and slides, the command that produced it, and where the numbers are stored.
Setup first: `bash scripts/pod/setup.sh` (see `scripts/README.md`). The data (`data/`, `dev/`) is not redistributable;
the public task files and the dev splits (`python -m sinmmlu.build_dev`) must be placed locally. Dated entries in
`docs/log.md` give the decisions and numbers of each round; raw pod outputs are in `results/runpod/<round>/`
(gitignored).

## Final system

| Result | Command | Stored in |
|---|---|---|
| LoRA training, validation accuracy 44.4 -> 54.8 (807 steps, 8.8 h) | `python -m sinmmlu.combined` then `bash scripts/final/train_qwen_combined_1028.sh` | `lora_adapters/qwen3.5-4b_lora_combined-si_epoch1` (md5 in `MANIFEST.md`), `docs/log.md` 2026-10-28 |
| Test predictions, Macro Avg 57.58 (Easy 54.67, Medium 52.30, Hard 65.77) | `bash scripts/final/think_1028.sh test` (+ `think_rest_1028.sh`) or `final_submission/team-sinbotics/inference.py` | `final_submission/team-sinbotics/predictions.jsonl`; leaderboard score from Codabench |
| 59.7% on screening set 2 (300 test-question subset, graded outside the code) | `bash scripts/final/think_1028.sh subset`, scored with `python -m sinmmlu.score_subset` | `docs/log.md` 2026-10-28 |

## Experiments by phase (report Section 5)

| Phase | Result | Script / module | Log entry |
|---|---|---|---|
| 1 | 1B models all at chance (25.6-26.6%) | pipeline runs on Colab/Kaggle; no pod script | `results/phase0_1b/results.csv`, `docs/log.md` 2026-10-15 |
| 2 | 59 conditions on screening set 1 (280 Q), Gemma-3-4B-it | `scripts/experiments/run_subset.sh wave1 wave2 wave3 report` | `docs/log.md` 2026-10-24 (waves 1-3, combinations) |
| 2 | cot_en_role 38.2 -> 45.6% on the 1,483-question confirmation set (+7.4, Holm p = 1.2e-6) | `scripts/experiments/confirm2.sh`, `select_and_test.sh` | `docs/log.md` 2026-10-24 "confirmation" |
| 2 | Answer pre-fill (trailing space) pitfall | `scripts/experiments/probe_prefill.sh` | `docs/log.md` 2026-10-24 |
| 2 | Submission 2 (1,024 tokens), exact reproduction 2,238/2,238 | `scripts/experiments/rerun_1026.sh` | `docs/log.md` 2026-10-26 |
| 2 | Negation chain (-1.4), option-order averaging (+0.7), 3-prompt ensemble (+1.8): none met the +2 rule | `negchain_1026.sh`, `last2_1026.sh` (in `scripts/experiments/`) | `docs/log.md` 2026-10-26 |
| 2 | Translation (NLLB, MADLAD, self-translation) and question-type handling | `python -m sinmmlu.translate`, `qtypes`, `negation` via `run_subset.sh` | `docs/log.md` 2026-10-24 |
| 3 | 6 Gemma LoRA adapters (Global-MMLU 40.1 -> 41.5; sinmulu_si 40.9 -> 47.1) | `python -m sinmmlu.train_lora`, `scripts/experiments/lora_1027.sh`, `loraB_1027.sh`, `train_r8_1027.sh` | `docs/log.md` 2026-10-27 |
| 3 | Adapters vs base on screening set 2 (50.3 / 49.3 vs 47.3) | `scripts/experiments/test300_1027.sh`, `test300b_1027.sh`, `test300c_1027.sh`, `cot_grid_1027.sh` | `docs/log.md` 2026-10-27 |
| 4 | Gemma -> Qwen3.5-4B +7.4; adapter 55.0 vs 54.7; thinking +4.7; final vs submission 2 +12.4 | `scripts/final/think_1028.sh`, `python -m sinmmlu.evaluate`, paired tests with `python -m sinmmlu.report` | `docs/log.md` 2026-10-27, 2026-10-28 |
| 4 | Subject-aware prompt lost 12 points | prompt `cot_en_role_subject_aware` in `sinmmlu/prompts.py` | `docs/log.md` 2026-10-27 |

## Data and checks

| Item | Command | Stored in |
|---|---|---|
| Train/test contamination check (18 public items, 165 sinmulu_si items removed) | `python -m sinmmlu.dedup` | `results/dedup_train_vs_test*.txt`, `MANIFEST.md` |
| Parameter counts (8B budget) | `python scripts/pod/count_params.py` | `MANIFEST.md`, `docs/log.md` 2026-10-24 |
| Dev splits | `python -m sinmmlu.build_dev` | `configs/subsets/` |
| Test subset of 300 and its scoring | `python -m sinmmlu.test_subset`, `score_subset` | `docs/log.md` 2026-10-27 |

## Caveats
- Raw per-question outputs of some 24-25 Oct runs were lost when a pod was terminated; only their numbers survive in
  `docs/log.md`. The 26 Oct rerun reproduced submission 2 exactly.
- Screening set 2 consists of public test questions; its answers were prepared manually with AI help and never used
  for training. Scores on it are optimistic (about +/-5.7 points).
- Experiment ids follow `<model>__<quant>__<prompt>__chat<on|off>__p<perms>__c<calibrate>__tr<translator>[__mt<N>]__<data>`;
  each pod round has its own `results.csv` under `results/runpod/<round>/` (gitignored); the numbers are also in `docs/log.md`.
