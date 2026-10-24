# Student Comment — 2026-10-14 16:24

To: Supervisor. Covers the work after `student_comment_2026-10-14_15-00.md` (your 13:36 note was answered there). Nothing is committed yet.

## 1. What we did

### 1.1 Evaluation code (Colab-ready, copied to the team Drive)
- `src/eval_logprob.py`: zero-shot log-prob scoring. Score the digits 1–4 after "පිළිතුර:" (both `"1"` and `" 1"` token variants, assert each is a single token), argmax = answer. Options: `--perms 4` (cyclic permutation averaging, mapped back to original options), `--calibrate` (subtract the log-prob prior from a content-free "N/A" question with four "N/A" options), `--chat on|off`, `--prompt subject_si|subject_en|nosubject`, `--quant`.
- Prompt = Appendix B adapted to 1–4 options, with the **Sinhala `subject_original`** (your B-Q1). ZWNJ removed, ZWJ kept.
- Chat template: single user message, no system prompt, no few-shot. Qwen3 thinking off. The Llama template injects today's date, so the date is fixed for reproducibility. fp32 for Gemma (fp16 overflows, no bf16 on T4), fp16 for the others.
- Dev runs print accuracy with a 95% interval and the predicted-option share, and write a row to `results/results.csv`. Test runs (`--data test`) write a validated `predictions.zip` and print **no** diagnostics.
- `tests/test_eval_core.py` is an offline test with an oracle scorer: permutation mapping, bias removal and calibration are checked.
- `src/run_grid.py`: experiment grid with a dry run. It skips experiments already in `results.csv` and loads each model once.
- `src/subject_report.py`: per-model subject-level (= per JSON file) accuracy from saved probabilities. Output: `results/subject_accuracy/<Model>.csv`.
- `colab_eval.ipynb` runs everything from Drive.
- **Eval data:** only `dev/dev_easy_full.jsonl` (all 1,851 public train questions). The public test set is used only to write predictions. `results.csv` now has only `dev_easy` + `ci95` (the Medium/Hard columns were removed, since we have no such data).

### 1.2 Model checks
Chat capability was **verified from the HF model cards**, not guessed from names. All three are instruction-tuned and document a chat template with message roles:
- `google/gemma-3-1b-it`
- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen3-0.6B` (post-trained; a separate `-Base` model exists)

The code stops if a config says chat-capable but the tokenizer has no chat template, and refuses `chat=on` for non-capable models.

### 1.3 Results so far (Easy dev, n = 1,851, 95% CI ≈ ±2 pp, chat on, perms 1, calib off, full precision)

| Model | Easy acc | Predicted option 1 | Fertility (tokens/word) |
|---|---|---|---|
| Qwen3-0.6B | 25.7% | 99.9% | 8.8 |
| Gemma-3-1B-it | 25.3% | 79% | 2.9 |
| Llama-3.2-1B-Instruct | 25.8% | 99.4% | 10.9 |
| Qwen3-0.6B, perms 4 | 23.7% | ~25% each | n/a |

Reading: all are at chance (always answering "3" would give 27.4%). Qwen and Llama answer "1" almost always, Gemma 79% of the time. Permutation averaging removes the bias but accuracy stays at chance, so there is no signal underneath. Gemma's tokenizer is far better for Sinhala (2.9 vs 9–11).

### 1.4 In progress
The grid for all three models (full precision): chat on/off × {calib off perms 1, calib off perms 4, calib on perms 1} = 18 experiments, 3 of which are done above. The remaining 15 run on Colab now. Results will be in the next comment.

### 1.5 One data-quality note
For one run (Qwen3 perms 4) probabilities were saved rounded to 6 decimals, which flipped one near-tied prediction (23.72% in `results.csv` vs 23.66% in the subject file). Fixed in the code (full precision). Not a conclusion-changing difference.

## 2. Questions

**Q1. Are 1B-class models worth the grid?** They are at chance, so differences between chat on/off, perms and calibration will be inside the ±2 pp interval. I'm running the grid anyway because the team lead asked for it. After it, should we stop here and go to the Phase A 3–4B / 7–8B models, or is there anything in these small-model results you want us to check?

**Q2. Decision rule for chat on/off, perms and calibration.** With CI ≈ ±2 pp, what difference counts as a real winner on the full Easy dev? Should I use a paired test on the same 1,851 questions (McNemar) instead of comparing intervals?

**Q3. Sanity check at 7B.** You asked for Qwen2.5-7B-Instruct at 25–30% as the pipeline check. On a Colab T4 this needs 4-bit. Our 1B grid is full precision, as the team lead asked. Is 4-bit acceptable for 7–8B screening (as in your §3), and is the 7B check the first thing to run before any other model?

**Q4. Calibration prior.** I use one content-free input ("N/A" question and options). Is this enough, or do you recommend averaging several content-free inputs (e.g. "N/A", "", "[MASK]") to get a more stable prior?

**Q5. Subject ablation timing.** The subject-name ablation (none / English / Sinhala) is meaningless on a chance-level model. Run it on the best 7–8B model, as you suggested ("one model, full Easy dev")?

**Q6. Noisy per-subject numbers.** Each JSON file has 100–164 questions (±8 pp or more). For the report, should we also pool subjects by category (humanities / social science / language / stem) for a less noisy view?

**Q7. First Codabench submission (due 19 Oct).** If the best 1B system is also at chance on dev, is it still worth spending one submission as a pipeline test, or should we wait for the best 7–8B zero-shot result?

**Q8. Version pinning.** We use Colab. I plan to pin `torch`, `transformers` and `accelerate` to the versions printed by Colab (the organisers will rebuild from `requirements.txt`). Is that acceptable instead of Kaggle versions?

— Student
