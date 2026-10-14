# MANIFEST (update as you go)

## Models used so far (screening only; none is part of a final submission yet)
| Model | Params | HF revision | Chat capable (model card) | Use |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | ~0.6B | recorded per run in `outputs/*/config.json` (`model_revision_resolved`); pin in configs | yes | Phase-0 pipeline test |
| google/gemma-3-1b-it | ~1.0B | same | yes | Phase-0 pipeline test |
| meta-llama/Llama-3.2-1B-Instruct | ~1.2B | same | yes | Phase-0 pipeline test |

Total at inference per system: one model at a time (limit 8B).
Environment (python / torch / transformers / accelerate / CUDA / GPU) is stored in every run's `config.json`; `requirements-lock.txt` comes from `pip freeze` on the Colab runtime that produced the results.

## Datasets
| Dataset | Use | Licence/permission | Notes |
|---|---|---|---|
| Public train (1851 Q, Easy, answers) | **evaluation only** (never trained on) | task data | 1 label fixed (Geography q_no 64 -> 3); 18 items have near-duplicates in test (`overlaps_test` flag in dev) |
| Public test (2238 Q, no answers) | prediction only | no redistribution | never tuned on, no diagnostics |
| naist-nlp/SinhalaMMLU (HF) | **NOT ACCESSED** (may contain shared-task test items, with answers) | CC BY-NC-ND 4.0, gated | deliberately not requested or downloaded |

## Dedup (src/dedup.py, see dedup_report.txt)
Method: char 5-gram Jaccard on norm_dedup() (punctuation/blank-runs/ZWJ/ZWNJ removed), question + sorted options; question-only as a second signal. Both Jaccards are computed exactly for every pair (fixed 2026-10-14 after the 16:35 review). REMOVE full >= 0.5; REVIEW full 0.35-0.5 or (q >= 0.5 and full >= 0.2); IGNORED = generic stems (q >= 0.5, full < 0.2). Stricter thresholds for collected past papers (`--source collected`).

| Candidates | vs | Remove | Review | Ignored (generic stems) |
|---|---|---|---|---|
| public train (1851) | public test (2238) | 18 pairs / 18 train items | 100 pairs | 21 pairs |

Public train is eval-only, so nothing was removed. Eval sets: `dev_easy_full` (n=1851, primary for zero-shot) and `dev_easy_clean` (n=1833, `overlaps_test` false) once training data from past papers is used. Embedding check (LaBSE) not run yet (needs a model download, pending approval).

## Known invalid results (quarantined)
Runs made before 2026-10-14 16:35 with `chat: on` in YAML were silently run with chat OFF (YAML parsed `on` as a boolean). They are recorded as `chat=True` in `results.csv` and are moved to `outputs/_invalid_chat_bug/` and `results/invalid_rows.csv` by `src/purge_invalid.py`.

## Closed-API use
None.
