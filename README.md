# SinMULU-JR: Sinhala MMLU shared task (ICTer 2026)

Base model: **google/gemma-3-4b-it**. Task rules: ≤ 8B total parameters at inference (every component, LoRA
included), open weights released before 2026-10-01, no retrieval, closed APIs only for generating training data
(disclosed), nothing tuned on test questions. Metric: macro average of Easy / Medium / Hard accuracy.

## Layout
```
sinmmlu/            the code (run every module with python -m sinmmlu.<module>)
  data.py           loading, normalisation (keep ZWJ, drop ZWNJ), uid, option permutation
  prompts.py        prompt registry: one name = one exact prompt text
  scoring.py        option-order averaging + contextual calibration on any label scorer
  model.py          model loading (bf16/fp32, NF4 4-bit; never fp16), label / cloze scoring, greedy generation
  evaluate.py       run a model under one or more conditions on dev or test -> outputs/, results/results.csv
  report.py         paired comparison of finished experiments (McNemar + Holm, bootstrap), per-subject table
  ensemble.py       average the probabilities of finished experiments (prompt ensembles)
  translate.py      Sinhala -> English (NLLB, MADLAD, self-translation), cached in cache/translations/
  submit.py         Codabench predictions.jsonl / zip + format validation
  build_dev.py      data validation + dev files (easy_full, easy_clean, inference_subset, fewshot_pool, inference_rest)
  dedup.py          contamination check against the test set
  tokens.py         tokenizer-only: fertility, prompt lengths
  debug.py          next-token inspection on dev questions
configs/models/     one YAML per model
configs/subsets/    uid lists of dev/inference_subset, fewshot_pool, inference_rest (tracked; dev/ is not)
runpod/             setup, model download, the screening waves, packing (see runpod/README.md)
tests/              offline tests (no model): python -m pytest -q
results/            results.csv (one row per dev experiment), dedup report
docs/log.md         lab notebook: decisions and results, newest first
archive/            earlier student code, phase-0 (≤1B) results, old supervisor/student notes
data/ dev/ outputs/ cache/ gitignored (the test licence forbids redistribution)
```

## Workflow
```bash
pip install -r requirements.txt                 # plus torch for your CUDA version
python -m sinmmlu.build_dev                     # validate data, write dev/*.jsonl
python -m pytest -q
python -m sinmmlu.evaluate --model configs/models/gemma3_4b_it.yaml --data dev/easy_full.jsonl --limit 16   # smoke test
python -m sinmmlu.evaluate --model configs/models/gemma3_4b_it.yaml --data dev/easy_full.jsonl \
    --prompt paper_nosubject paper_subject_si --perms 1 4 --calibrate false true
python -m sinmmlu.report --data dev/easy_full.jsonl --baseline <exp_id>
python -m sinmmlu.evaluate --model configs/models/gemma3_4b_it.yaml --data test --prompt <best> --perms <best>
```
Experiment id: `<model>__<quant>__<prompt>__chat<on|off>__p<perms>__c<calibrate>__tr<translator>__<data>`. Finished experiments are
skipped; `--force` reruns. On Windows add `-X utf8` (Sinhala output).

## Rules this code enforces
- Dev = public train (Easy only). Test rows are only scored to write predictions; test runs print no accuracy-like
  diagnostics (the label mass is printed as a validity check only).
- Prompts are never truncated (a too-long prompt raises). Non-finite logits raise.
- Every run stores the model revision, total parameter count, resolved dtype, library versions and git revision.
