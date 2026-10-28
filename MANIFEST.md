# MANIFEST (models, data and closed-API use; goes into the code package and the report)

## Models
| Model | Total params | HF revision | Use |
|---|---|---|---|
| **Qwen/Qwen3.5-4B** | 4,659,865,088 checkpoint (text model 4,205,751,296, loaded as Qwen3_5ForCausalLM; vision tower not loaded) | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (released 2026-02-27, Apache-2.0) | **base model of the final system** |
| **LoRA adapter, combined data** (`SinhalaMMLU-SharedTask/team-sinbotics`, `adapter/`) | 30,474,240 (rank 16), merged into the weights | adapter md5 6d42bd99433fbef9b44b48245cdc68cc | **final system**; total at inference 4.24B |
| google/gemma-3-4b-it | 4,300,079,472 (text 3,880,263,168 + vision 419,816,304; counted from the safetensors) | 093f9f388b31de276ce2de164bdc2081324b9767 | base model of submissions 1 and 2, and of the Gemma LoRA runs |
| google/gemma-3-1b-it, Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B-Instruct | 1.0B / 0.6B / 1.2B | see archive/phase0_1b/outputs | phase-0 pipeline tests only (all at chance) |
| facebook/nllb-200-distilled-600M | 615,073,792 (tied) | f8d333a098d19b4fd9a8b18f94170487ad3f821d | screening only: Sinhala→English; CC-BY-NC 4.0 |
| google/madlad400-3b-mt | 2,940,374,016 | fa184c675da0b5c9e1c8694fccd4e12e2d422094 | screening only: Sinhala→English; Apache 2.0 |
| google/gemma-3-4b-pt | 4,300,079,472 | see outputs/*/config.json | screening reference only |

Inference budget of the final system: Qwen3.5-4B text model + LoRA = 4.24B (limit 8B); one model, no other component.

## Datasets
| Dataset | Use | Licence / permission | Notes |
|---|---|---|---|
| Public train (1,851 Q, Easy, answers) | evaluation (dev) only | task data | Geography q_no 64 label fixed to 3; 18 items near-duplicate test items (`overlaps_test`) |
| Public test (2,238 Q, no answers) | predictions only | no redistribution | never trained on |
| Global-MMLU, Sinhala (`CohereLabs/Global-MMLU`, `si`, rev 0e619dbeb34206cd48705a1a0ea7fb21cae09993) | LoRA training | Apache-2.0 | 14,324 Q (3 with an empty option dropped); 12,926 train / 1,398 val |
| sinmulu_si (our collection) | LoRA training | ours; published separately | Built from 3,486 Sri Lankan exam-paper PDFs (O/L, A/L, provincial term tests); 15,355 Q → 15,190 after removing 165 overlaps with the test set (manual review of TF-IDF / difflib candidates) → 15,180 after removing 10 duplicates; 13,749 train / 1,431 val |
| naist-nlp/SinhalaMMLU (HF) | **not accessed** | CC BY-NC-ND 4.0, gated | its card says the shared-task Hard test set is built from it |

## Dedup (`python -m sinmmlu.dedup`, results/dedup_train_vs_test.txt)
Character 5-gram Jaccard on normalised text; full = question + sorted options, q = question only.
Public / generated: remove full ≥ 0.5, review 0.35–0.5. Collected past papers: remove ≥ 0.35, review 0.25–0.35.

| Candidates | vs | Remove | Review | Ignored (generic stems) |
|---|---|---|---|---|
| public train (1,851) | public test (2,238) | 18 | 100 | 21 |
| sinmulu_si (15,355) | public test (2,238) | 165 (143 + 22, manual review) | 505 pairs ≥ 0.6 (round 1), ≥ 0.7 (round 2) | 260 test Q still > 0.7 on question similarity, judged different template questions |

## Closed-API use
| API | Use | Where disclosed |
|---|---|---|
| Claude Sonnet 5.5 (Anthropic) | creating training data only: transcribing the page images of the sinmulu_si exam papers, and labelling answers where no official marking scheme existed (kept only when highly confident) | submission README, this manifest, the report |

No closed API, LLM service or retrieval is used at inference.
