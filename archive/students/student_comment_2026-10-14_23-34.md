# Student Comment — 2026-10-14 23:34

To: Supervisor. Covers everything since `student_comment_2026-10-14_16-42.md` (your 16:35 review was answered there). Nothing is committed yet.

## 1. Done

**Your review items (confirmed on real data)**
- Token-length check on real tokenizers (eval set, n = 1,851; test n = 2,238 for lengths only): prompts over 1024 tokens: **Qwen3-0.6B 9 (chat off) / 10 (chat on) dev, 15 test; Llama-3.2-1B 19 / 24 dev, 64 / 78 test; Gemma-3-1B 0.** Longest prompt: Qwen 1,284, Llama 1,586, Gemma 481 tokens. So B2 was real for Qwen and Llama; Gemma was never affected. This also answers my question D1: Llama's old rows were not safe.
- Old Qwen/Llama rows were quarantined; **the team lead then decided to throw away all previous results and rerun everything fresh** with the fixed code.
- Truncation is gone. I also **removed my own arbitrary 4096 token cap and the `--max-length` option**: the only limit is the model's real context length (about 32k for Qwen3-0.6B and Gemma-3-1B-it, 128k for Llama-3.2-1B; confirmed on the model cards by the team lead). A longer prompt raises an error.

**Changes requested by the team lead**
- **Prompt is now `nosubject`** (no subject name) in all configs. The prompt is part of every experiment name and subject-file column, so other prompts can be added later without overwriting.
- **Tokeniser fertility** is computed over all 1,851 eval questions, printed when each model loads, stored in `config.json` and `results.csv` notes, and written to `results/fertility.csv` by the histogram script. Qwen3-0.6B: **8.66 tokens/word** (the first 1,000-question figure was 8.78).
- **Token-length histogram** (`src/token_hist.py`, 1 × 3 grid, shared axes, median / p99 / max lines, tokenizer only, eval set only).
- **Validity check `digit_mass`**: the average total probability the model puts on the four digit tokens at the scored position. New column in `results.csv`, printed per run, and a WARNING below 0.5.
- Run recording of time was added and then removed on request.
- Notebook: fresh-start cell (explicit `CONFIRM = True`), a `CONFIGS` variable to choose which models to run (now Qwen only), grid, results and subject-accuracy cells, Colab version pinning cell (`requirements-lock.txt`).
- Per-run storage now includes python / torch / transformers / CUDA / GPU, the resolved chat mode, the longest prompt and the HF revision lookup.
- `MANIFEST.md`: models, dedup counts, quarantine note updated.

**Checks (dedup, B3)** unchanged: REMOVE 18, REVIEW 100, IGNORED 21; 18 eval items flagged `overlaps_test` (dev_easy_clean n = 1,833).

**Our laptop** (for reference): RTX 4050 Laptop 6 GB, 32 GB RAM. OK for 0.6–1B models and debugging; not for 7–8B (needs ≥ 6 GB for weights alone in 4-bit). Colab T4 stays the main platform.

## 2. Results so far (Qwen3-0.6B only, nosubject prompt, Easy dev n = 1,851, full precision, new code)

| chat | calibration | option orders | Easy acc | 95% CI | digit mass |
|---|---|---|---|---|---|
| on | no | 1 | 25.61% | ±2.0 | **0.028** |
| on | no | 4 | 24.64% | ±2.0 | 0.029 |
| on | yes | 1 | 26.04% | ±2.0 | 0.028 |
| off | no | 1 | 25.55% | ±2.0 | **0.000** |
| off | no | 4 | 23.18% | ±1.9 | 0.000 |
| off | yes | 1 | 26.36% | ±2.0 | 0.000 |

Reading: all six are at chance (always answering "3" would give 27.4%). The new validity check explains why: **at the scored position Qwen puts only 2.8% of its probability on the digits with chat on, and about 0% with chat off.** The earlier "always answers option 1" finding is therefore a consequence of reading a position where the model does not want to emit a digit. Gemma and Llama were not run in this fresh run (CONFIGS = Qwen only).

**Debug notebook** (`colab_debug_next_token.ipynb`): first 20 dev questions, one at a time, with exact input text, last input tokens, top-15 next tokens, digit-candidate probabilities (`'1'` vs `' 1'`, including multi-token), and a 100-token greedy continuation. Qwen3-0.6B **generates other text, not an answer digit**, with chat on and with chat off (observation from the team lead's runs).

## 3. Questions for the supervisor

**Q1. How should we get a usable answer from a model that does not start with a digit?** Options I see; please rank or reject:
  1. **Prefill the answer** in chat mode (assistant turn starts with "පිළිතුර:" so the next token is the digit);
  2. **Score the two-token answer** (space + digit) when the tokenizer splits `' 1'` (Qwen does);
  3. an explicit instruction line ("reply with only the number");
  4. **few-shot** examples to teach the format (paper: little accuracy gain, but may fix format; makes prompts much longer for Qwen / Llama);
  5. **cloze scoring** (log-likelihood of each option's text, length-normalised) instead of option digits.
  Which first, and are 1–4 acceptable under the task rules (no retrieval, no manual step at inference)?

**Q2. Is a change of prompt format allowed to be chosen on the Easy dev set?** I plan to compare formats only on public train (never test), report all of them, and freeze one before 7–8B screening.

**Q3. Is `digit_mass` a sound validity criterion?** I propose: report it for every run, and only compare models or settings whose mean digit mass is ≥ 0.5 (otherwise the argmax over digits is weak evidence). Do you agree, and is 0.5 a sensible threshold?

**Q4. Do 0.6–1B models still matter?** Qwen is at chance in every setting. The team lead wants Gemma-3-1B and Llama-3.2-1B run as well. After the format issue is fixed, should we stop at the 1B models or go straight to Phase A (3–4B / 7–8B in 4-bit, Colab T4)?

**Q5. The same digit-mass check on test runs:** the run prints the mean and minimum digit mass for test predictions too. It is a validity check, not a prediction distribution. Is that acceptable under "never look at test diagnostics"?

**Q6. Still open from my 16:42 comment:** D1 (resolved by the length numbers above), D2 (per-level columns kept out of `results.csv`; extra levels go to `notes`), D4 (organiser email: team lead decided not to send), commit permission (nothing committed).

## 4. Not done yet
- Gemma-3-1B and Llama-3.2-1B in the fresh run, and the subject-name ablation.
- `src/compare.py` (McNemar + bootstrap), `configs/domain_map.yaml`, LaBSE dedup, Qwen2.5-7B-Instruct 4-bit sanity check, Codabench format test.
- Pinned `requirements.txt` (the lock file is produced by the notebook on Colab).

— Student
