# MANIFEST (keep up to date; it goes into the code package and the report)

## Models
| Model | Total params | HF revision | Use |
|---|---|---|---|
| google/gemma-3-4b-it | 4,300,079,472 (text 3,880,263,168 + vision 419,816,304; counted from the safetensors) | 093f9f388b31de276ce2de164bdc2081324b9767 | base model of submission 1 |
| google/gemma-3-1b-it, Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B-Instruct | 1.0B / 0.6B / 1.2B | see archive/phase0_1b/outputs | phase-0 pipeline tests only (all at chance) |
| facebook/nllb-200-distilled-600M | 615,073,792 (tied) | f8d333a098d19b4fd9a8b18f94170487ad3f821d | screening only (D2): Sinhala→English; CC-BY-NC 4.0 |
| google/madlad400-3b-mt | 2,940,374,016 | fa184c675da0b5c9e1c8694fccd4e12e2d422094 | screening only (D3): Sinhala→English; Apache 2.0 |
| google/gemma-3-4b-pt | 4,300,079,472 | see outputs/*/config.json | screening reference only (C9) |

Inference budget of submission 1: Gemma-3-4B-it only = 4.30B (limit 8B). Gemma + NLLB would be 4.92B, Gemma + MADLAD 7.24B (both allowed).

## Datasets
| Dataset | Use | Licence / permission | Notes |
|---|---|---|---|
| Public train (1,851 Q, Easy, answers) | evaluation (dev) | task data | Geography q_no 64 label fixed to 3; 18 items near-duplicate test items (`overlaps_test`) |
| Public test (2,238 Q, no answers) | predictions only | no redistribution | never tuned on |
| naist-nlp/SinhalaMMLU (HF) | **not accessed** | CC BY-NC-ND 4.0, gated | its card says the shared-task Hard test set is built from it |

## Dedup (`python -m sinmmlu.dedup`, results/dedup_train_vs_test.txt)
Character 5-gram Jaccard on normalised text; full = question + sorted options, q = question only.
Public / generated: remove full ≥ 0.5, review 0.35–0.5. Collected past papers: remove ≥ 0.35, review 0.25–0.35.

| Candidates | vs | Remove | Review | Ignored (generic stems) |
|---|---|---|---|---|
| public train (1,851) | public test (2,238) | 18 | 100 | 21 |

## Closed-API use
None.
