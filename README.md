# SinMULU-JR — Sinhala MMLU shared task (ICTer 2026)

Constraints: ≤ 8B total params at inference, open weights released before 2026-10-01, no retrieval, closed APIs only for training-data generation (disclosed).

```
data/      public train (answers, Easy only) + public test (no answers, gitignored)
src/       data_utils.py, validate_data.py, build_dev.py, eval_logprob.py*, train_sft.py*, predict.py, (dedup.py: section 2.3, TODO)
dev/       dev_easy_full.jsonl (all 1851 public train), dev_easy_screen.jsonl (~150, stratified)
configs/   one YAML per experiment
results/   results.csv (one row per experiment)
comments/  supervisor_comments_*, student_comment_*
```
`*` = stub, planned.

Quick start: `python -X utf8 src/validate_data.py && python -X utf8 src/build_dev.py`

**Team rule:** public train is used for EVALUATION ONLY (no training) until the organisers say otherwise.

## Evaluating models (Gemma-3-1B-it, Llama-3.2-1B-Instruct, Qwen3-0.6B)
Run on Colab with `colab_eval.ipynb` (sections 1-6). Eval set = public train (Easy only, never trained on); public test is used only to write predictions.

Command line (any machine with a GPU):
```bash
pip install -r requirements.txt
huggingface-cli login            # Gemma and Llama are gated: accept the licence on the model page first
python -X utf8 tests/test_eval_core.py                       # offline logic tests
python -X utf8 src/run_grid.py --dry-run                      # what is done / still to run
python -X utf8 src/run_grid.py                                # 18 experiments, skips finished ones
python -X utf8 src/subject_report.py                          # per-model subject accuracy CSVs
python -X utf8 src/eval_logprob.py --config configs/gemma3_1b.yaml --data test --chat on --exp-id gemma3_1b_chaton   # test predictions
```
- Prompt: `nosubject` by default (configs); `--prompt subject_si|subject_en` overrides. The prompt is part of every experiment name.
- Grid per model: chat on/off x {calib off perms 1, calib off perms 4, calib on perms 1}, full precision. Chat capability is set in each config (`chat_capable`, checked on the model card) and verified against the tokenizer's chat template.
- Chat mode is always `on` or `off` (quote it in YAML; `auto` = on only if chat-capable). The resolved mode is stored in every run.
- Prompts are never truncated; a prompt above the model context length raises an error. `src/token_hist.py` plots the prompt token-length distribution per model (tokenizer only).
- Validity check: `digit_mass` = average total probability on the four answer digits (warning below 0.5).
- Dev runs print accuracy (+95% CI) and predicted-option share, and append a row to `results/results.csv`. Test runs print no diagnostics.
- Every run stores library versions, the longest prompt and the resolved HF revision in `outputs/<exp_id>/config.json`.
- `src/dedup.py` checks candidate data against the test set; `src/purge_invalid.py` quarantines untrustworthy rows (only needed for partial cleanups).
