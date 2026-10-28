# SinMULU-JR: Team SinBotics' system for the Sinhala MMLU Shared Task (ICTer 2026)

**Final system:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) + our LoRA adapter (rank 16, one epoch on Global-MMLU Sinhala + sinmulu_si), with English teacher-role reasoning in Qwen's thinking mode. It has 4.24B parameters at inference.
Submission repo: [SinhalaMMLU-SharedTask/team-sinbotics](https://huggingface.co/SinhalaMMLU-SharedTask/team-sinbotics). Its mirror is in [`final_submission/team-sinbotics/`](final_submission/team-sinbotics/), without the weights.

Task rules:
- ≤ 8B parameters at inference, counting every component (LoRA included);
- open weights released before 2026-10-01;
- no retrieval;
- closed APIs only for creating training data, and disclosed;
- nothing tuned on test questions.

Metric: macro average of Easy / Medium / Hard accuracy.

## System
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
runpod/                  pod setup, model download, the final training / inference scripts (see runpod/README.md)
runpod/experiments/      the scripts of the earlier experiment rounds (dated)
final_submission/        mirror of the Hugging Face submission repo (inference.py, README, training code; no weights)
tests/                   offline tests (no model): python -m pytest -q
results/                 results.csv (one row per dev experiment), dedup report
report/                  system description report (LaTeX)
docs/log.md              lab notebook: decisions and results, newest first
archive/                 earlier student code, phase-0 (≤ 1B) results, old notes
gitignored               data/ dev/ outputs/ cache/ models/ lora_adapters/ logs/ submissions/ runpod_results*/
                         (the task data may not be redistributed; weights and raw outputs are too large)
```

## Reproducing the final system
```bash
pip install -r requirements.txt          # torch 2.8 separately; then: pip install causal-conv1d==1.7.0 --no-build-isolation
python -m pytest -q
python -m sinmmlu.combined               # training data (sinmulu_si: data/sinmulu_si/sinmulu_si.json, published separately)
bash runpod/train_qwen_combined_1028.sh  # Qwen3.5-4B download + LoRA training -> models/qwen35_4b_lora_combined/
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
