# Training and data preparation (Team SinBotics)

All commands run from this folder (`training/`), as `python -m sinmmlu.<module>`. Python 3.12, torch 2.8.0 (CUDA 12.8), then:
```bash
pip install -r requirements.txt
pip install causal-conv1d==1.7.0 --no-build-isolation
```

## 1. Data
| Step | Command | Output |
|---|---|---|
| Global-MMLU Sinhala: download a fixed revision (`CohereLabs/Global-MMLU`, `si`, rev `0e619dbe…`), drop 3 questions with an empty option, convert to the competition format, ids `gsi…`, 10% validation per subject (seed 42) | `python -m sinmmlu.gmmlu --val-fraction 0.10` | `data/global_mmlu_si/competition_format_val10/{train,val}.json` (12,926 / 1,398) |
| sinmulu_si (our collection, 15,190 questions; expected at `data/sinmulu_si/sinmulu_si.json`): remove 10 duplicates (same question and options), 10% validation per subject (seed 42) | `python -m sinmmlu.sinmulu --val-fraction 0.10` | `data/sinmulu_si/competition_format_val10/{train,val}.json` (13,749 / 1,431) |
| Combined training set (union of the two; any validation question whose text also appears in training moves to training, 0 here) | `python -m sinmmlu.combined` (runs the two steps above if needed) | `data/combined_si/competition_format_val10/{train,val}.json` (26,675 / 2,829) |

Fields: `q_no`, `subject`, `question`, `choices` (4), `answer` ("1"–"4").

**sinmulu_si is not included in this repo:** our team plans to publish it separately. It is available to the organisers on request. Without it, the Global-MMLU step runs as is, and the trained adapter is published (`../adapter/`), so inference does not depend on it.

## 2. LoRA fine-tuning
```bash
bash scripts/train_lora.sh
```
This is `python -m sinmmlu.train_lora --model configs/models/qwen35_4b.yaml --train data/combined_si/competition_format_val10/train.json --val data/combined_si/competition_format_val10/val.json --max-batch-tokens 12288 --out models/qwen35_4b_lora_combined`.

| | |
|---|---|
| Base | `Qwen/Qwen3.5-4B` rev `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16 |
| Objective | log P(gold digit) after the plain digit prompt `en_instr_nosubject`, with the chat reply pre-filled "Answer: "; loss on that one token only |
| LoRA | rank 16, alpha 32, dropout 0.05; q/k/v/o/gate/up/down projections and the linear-attention in_proj_qkv/in_proj_z/out_proj of the 32 text layers; 30,474,240 parameters |
| Optimiser | AdamW, lr 1e-4, cosine schedule, 3% warm-up, 32 questions per step, 1 epoch = 807 steps, seed 42; 876 training questions longer than 1,024 tokens skipped |
| Hardware / time | 1× RTX 3090 24 GB, 8.8 h |
| Validation (2,745) | 44.4% → 54.8% |

Outputs: `models/qwen35_4b_lora_combined/adapter/` (the files in `../adapter/`), the merged model, `train_config.json` and `train_log.jsonl`. To rebuild the merged model from the published adapter instead: `python -m sinmmlu.merge_lora --model configs/models/qwen35_4b.yaml --adapter <adapter folder> --out models/qwen35_4b_lora_combined`.

## 3. Final predictions with the package (what produced `../predictions.jsonl`)
```bash
python -m sinmmlu.evaluate --model configs/models/qwen35_4b_lora_combined.yaml --data test \
    --prompt cot_en_role_think --chat on --max-new-tokens 4096
```
`--data test` reads the test JSON files from `data/public test/{easy,medium,hard}/` (not included; the organisers' dataset). The run writes `outputs/<run id>/predictions.zip`. `logs/final_run_config.json` records our final run: settings, package versions, GPU and run time. For that run, the model config had `gen_batch_size: 48` (A100 40 GB). `../inference.py` is the self-contained equivalent, downloading everything from Hugging Face.

## Package contents (`sinmmlu/`)
- `data.py`: loading, text normalisation, splits
- `prompts.py`: all prompt variants (final: `cot_en_role_think`)
- `model.py`: model loading, chat rendering, batched generation and digit scoring
- `evaluate.py`: experiment runner
- `scoring.py`, `metrics.py`, `submit.py`: scoring, statistics, the submission file and its format check
- `gmmlu.py`, `sinmulu.py`, `combined.py`: data preparation
- `train_lora.py`, `merge_lora.py`: fine-tuning
- the other modules are earlier experiments (translation, negation, ensembles) and are not used by the final system
