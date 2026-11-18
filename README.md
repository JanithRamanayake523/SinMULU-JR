# SinMULU-JR: Team SinBotics' system for the Sinhala MMLU Shared Task (ICTer 2026)

**Research question:** can a small (<= 8B) open model answer Sinhala school-exam questions better if it reasons in English, and what else (translation, scoring, prompts, fine-tuning, thinking mode) changes the result? We ran about 70 configurations in four phases and report what helped and what did not, with the data, code and commands to reproduce each result.

**Main findings** (details in [`report/`](report/) and [`docs/log.md`](docs/log.md)):
- English reasoning over the untranslated Sinhala question was the largest inference-time gain on Gemma-3-4B-it (38.2 -> 45.6% on 1,483 held-out questions, +7.4 points).
- Real exam questions (our `sinmulu_si` collection) taught a LoRA adapter more than translated Global-MMLU (validation +6.2 vs +1.4 points), but the gain did not reliably carry over to reasoning prompts.
- Switching the base model to Qwen3.5-4B (+7.4) and turning on its thinking mode (+4.7) mattered most. Final system: Qwen3.5-4B + rank-16 LoRA, 4.24B parameters, Macro Avg 57.58 on the shared-task leaderboard (11th place).
- Most other ideas (translation, few-shot, option-order averaging, negation chains, ensembles) gave no confirmed gain.

Each reported number is mapped to its command and results file in [`docs/reproduce.md`](docs/reproduce.md).
Submission repo: [SinhalaMMLU-SharedTask/team-sinbotics](https://huggingface.co/SinhalaMMLU-SharedTask/team-sinbotics). Its mirror is in [`final_submission/team-sinbotics/`](final_submission/team-sinbotics/), without the weights.

## Shared-task setting
Sinhala MMLU Shared Task @ ICTer 2026. Rules:
- <= 8B parameters at inference, counting every component (LoRA included);
- open weights released before 2026-10-01;
- no retrieval;
- closed APIs only for creating training data, and disclosed;
- nothing tuned on test questions.

Metric: macro average of Easy / Medium / Hard accuracy.

## Final system
1. **Data.** Global-MMLU Sinhala and sinmulu_si (our collection of Sri Lankan exam questions; its overlap with the test set was removed) are combined, with a 10% validation set per subject: `sinmmlu/gmmlu.py`, `sinmulu.py`, `combined.py`.
2. **Fine-tuning.** LoRA rank 16 on Qwen3.5-4B. The loss is on the answer digit only, after a plain prompt pre-filled with "Answer: ". One epoch (807 steps) took 8.8 h on an RTX 3090. Validation accuracy went from 44.4% to 54.8%. Code: `sinmmlu/train_lora.py`, `merge_lora.py`.
3. **Inference.** The prompt `cot_en_role_think`: "You are an expert Sri Lankan teacher of {subject} … think step by step in English", with the Sinhala question as given. Qwen's thinking mode is on, decoding is greedy with at most 4,096 new tokens, and the answer is the argmax over the digit tokens after "Answer: ". Code: `sinmmlu/prompts.py`, `model.py`, `evaluate.py`; standalone version: `final_submission/team-sinbotics/inference.py`.

The full description (data, settings, results of all experiments) is in [`final_submission/team-sinbotics/README.md`](final_submission/team-sinbotics/README.md). The lab notebook is [`docs/log.md`](docs/log.md), and the model and data manifest is [`MANIFEST.md`](MANIFEST.md).

## Layout
```
sinmmlu/                 the package (run every module as python -m sinmmlu.<module>)
  data.py                loading, normalisation (keep ZWJ, drop ZWNJ), uid, splits, option permutation
  prompts.py             prompt registry: one name = one exact prompt text (final: cot_en_role_think)
  model.py               model loading, chat rendering (thinking switch), label / cloze scoring, batched generation
  evaluate.py            run a model under one or more conditions on dev or test -> outputs/, results/results.csv
  scoring.py metrics.py  option-order averaging, calibration, McNemar / bootstrap statistics
  report.py ensemble.py  paired comparison of finished experiments; probability ensembles
  submit.py              Codabench predictions.jsonl / zip + format validation
  gmmlu.py sinmulu.py combined.py   training data: Global-MMLU si, sinmulu_si, both (10% validation per subject)
  train_lora.py merge_lora.py       LoRA fine-tuning of the answer digit; rebuilding a merged model from an adapter
  build_dev.py dedup.py  dev files from public train; contamination check against the test set
  test_subset.py score_subset.py    stratified subset of the test questions and its scoring against a separate answer key
  translate.py negation.py qtypes.py rationales.py select_winner.py predict_winner.py tokens.py debug.py
                         earlier experiments and tools (translation, negation, question types, worked examples, ...)
configs/models/          one YAML per model (base models, merged LoRA models)
configs/subsets/         uid lists of the dev splits
scripts/final/           training and inference of the final system (see scripts/README.md)
scripts/pod/             GPU-pod setup, model download, speed test, packing results
scripts/experiments/     the scripts of the earlier experiment rounds (dated)
final_submission/        mirror of the Hugging Face submission repo (inference.py, README, training code; no weights)
tests/                   offline tests (no model): python -m pytest -q
report/                  submitted system description report (ACL LaTeX + PDF), appendices, slides, presenter notes
results/                 dedup report; phase-0 results; raw pod outputs (one results.csv per round)
results/phase0_1b/       results of the phase-0 (<= 1B models) pipeline tests
results/runpod/          raw pod outputs per round: 2026-10-24_25, 2026-10-26, 2026-10-27 (gitignored)
docs/log.md              lab notebook: decisions and results, newest first
docs/organiser/          organisers' report template (gitignored)
papers/                  notes on the SinhalaMMLU paper
MANIFEST.md              models, data and closed-API disclosure
gitignored               data/ dev/ outputs/ cache/ models/ lora_adapters/ logs/ predictions/ submissions/ results/runpod/
                         (the task data may not be redistributed; weights and raw outputs are too large)
```

## Reproducing the final system
```bash
pip install -r requirements.txt          # torch 2.8 separately; then: pip install causal-conv1d==1.7.0 --no-build-isolation
python -m pytest -q
python -m sinmmlu.combined               # training data (sinmulu_si.json from huggingface.co/datasets/JaniRaviRash-523/sinmulu-si in data/sinmulu_si/)
bash scripts/final/train_qwen_combined_1028.sh  # Qwen3.5-4B download + LoRA training -> models/qwen35_4b_lora_combined/
python -m sinmmlu.evaluate --model configs/models/qwen35_4b_lora_combined.yaml --data test \
    --prompt cot_en_role_think --chat on --max-new-tokens 4096
```
Without retraining, `python -m sinmmlu.merge_lora --model configs/models/qwen35_4b.yaml --adapter <adapter folder> --out models/qwen35_4b_lora_combined` rebuilds the merged model from the published adapter. The organisers' entry point is `final_submission/team-sinbotics/inference.py`. It downloads everything from Hugging Face and takes about 5 h for 2,238 questions on an A100 40 GB.

Experiment ids: `<model>__<quant>__<prompt>__chat<on|off>__p<perms>__c<calibrate>__tr<translator>[__mt<N>]__<data>`. Finished experiments are skipped; `--force` reruns them. On Windows, add `-X utf8` for Sinhala output.

## Rules this code enforces
- Prompts are never truncated: a prompt that is too long raises an error, and so do non-finite logits.
- Generation is greedy and deterministic. Every run stores the model revision, total parameter count, resolved dtype, library versions and git revision.
- Test rows are scored only to write predictions. Training data never contains test questions; sinmulu_si was checked against the test set first.

## Team SinBotics
Sachintha Weththasinghe (SLIIT), Hasitha Hirushan (SLIIT), Hasaan Saadhiq (SLIIT), Janith Ramanayake (UoM).

Dataset paper: Pramodya et al., *SinhalaMMLU: A Comprehensive Benchmark for Evaluating Multitask Language Understanding in Sinhala*, EMNLP 2025.
