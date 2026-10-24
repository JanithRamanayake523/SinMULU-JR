# Supervisor Comments — 2026-10-14 16:35

Re: `student_comment_2026-10-14_15-00.md` and `student_comment_2026-10-14_16-24.md`

Strong progress. The eval code is well structured, the offline oracle test is a good idea, the dedup report contains no test text, and you questioned the 1B results instead of over-reading them. I read every new file, ran `tests/test_eval_core.py` (it passes), and re-checked the dedup and prompt lengths against the data. **I found two bugs that affect results you have already reported**, plus one in the dedup. Details are in §B.

---

## A. Run approval

| Script | Status | Condition |
|---|---|---|
| `tests/test_eval_core.py`, `src/validate_data.py`, `src/build_dev.py`, `src/predict.py --validate` | ✅ **Approved now** | — |
| `src/dedup.py` | ⚠️ **Approved after fix B3** | Rerun and replace `dedup_report.txt` / MANIFEST counts |
| `src/eval_logprob.py`, `src/run_grid.py`, `colab_eval.ipynb` | ⛔ **Not approved until B1 + B2 are fixed** | Then run a 20-question smoke test and check the printed chat mode and the max token length |
| `src/subject_report.py` | ✅ After B1 (its labels inherit the same bug) | — |
| `train_sft.py` | n/a (stub) | — |

If the Colab grid is running right now, let it finish. The Gemma and Llama rows from the grid are valid, because `run_grid.py` sets the chat mode explicitly as a string. Treat the **Qwen rows as invalid** (B1).

---

## B. Code review: bugs, ranked

### B1. 🔴 Critical: `chat: on` in YAML silently turns chat **OFF**
PyYAML follows YAML 1.1, so `on` / `off` are parsed as the **booleans** `True` / `False`. I verified this: `yaml.safe_load("chat: on") → {'chat': True}`.

Trace in [eval_logprob.py:246](src/eval_logprob.py:246): `cfg["chat"]` is `True`, so the `else` branch calls `set_chat(True)`. Then `True == "on"` is `False`, so `state["chat"] = False`, and **the raw prompt is used without the chat template**. No error is raised.

Consequences:
- `results.csv` row 1 (`qwen3_0.6b_zs`, `chat=True`) and row 4 (`qwen3_0.6b_perm4`) were run **without** the chat template, but your 16:24 table reports them as "chat on."
- [run_grid.py:151](src/run_grid.py:151) `norm_chat("True") → "on"`, so the grid thinks Qwen chat-on (p1 and p4) is **done** and skips it. Those two experiments have never actually run.
- Any direct `eval_logprob.py --config configs/*.yaml` run today uses chat off.
- The Gemma and Llama rows say `chat=auto`, so they correctly ran with chat on.

Fix (all three):
1. In configs, quote it: `chat: "on"`, or better, use `chat: auto`.
2. In `load_cfg`, normalise: `if isinstance(cfg["chat"], bool): cfg["chat"] = "on" if cfg["chat"] else "off"`, then `assert cfg["chat"] in ("auto", "on", "off")`.
3. In `set_chat`, `assert mode in ("on", "off")`, so this class of bug fails loudly.
4. Print the resolved chat state in each run (`[run] chat=on/off`) and store the **resolved** value, not the config value, in `results.csv` and `config.json`.
5. Delete the two Qwen rows from `results.csv` (and their `outputs/` dirs), then rerun.

The same CLI issue applies to `--chat-capable`: its default is `None`, so argparse gives it `type=str`, and `--chat-capable false` becomes the truthy string `"false"`. Give it the same bool parser as the other bool flags.

### B2. 🔴 High: truncation cuts off the answer position on long prompts
[eval_logprob.py:266](src/eval_logprob.py:266) uses `truncation=True, max_length=1024`. The tokenizer's default `truncation_side` is **right**, so an over-long prompt loses its **end**: the options, `පිළිතුර:` and the assistant header. The logit you then read at position −1 is not the answer position, and the prediction is garbage, with no warning.

Is it real? The longest test prompt is **881 characters** (p99 is 691), measured with `build_prompt`. Your fertility numbers (Llama 10.9 and Qwen 8.8 tokens per word) mean these reach about 1.2–1.5k tokens before the chat template is added. So some Hard / long items **are** truncated for Llama and Qwen today. It will be worse for Medium/Hard dev data and any few-shot prompts.

Fix: set `truncation=False` and `max_length=4096`, and **assert** that every encoded length is ≤ the model context. Log `max_len` per run. Never truncate an MCQ prompt silently. If you ever must, drop few-shot examples, not the question.

### B3. 🟠 Medium: the dedup "IGNORED" bucket is mostly a threshold artefact
[dedup.py:63-71](src/dedup.py:63) computes both the `full` and the `q` matches with a cut-off at `thr=0.35`. Any `full` below 0.35 is **never stored**, so it reads as `0.0`. As a result:
- The REVIEW rule `q ≥ 0.5 and full ≥ 0.2` can never fire for `full` in [0.2, 0.35). It's dead code.
- Those pairs fall into IGNORED as `full < 0.2`. I recomputed the exact Jaccard for your 80 IGNORED pairs: **59 of them have full ≥ 0.2**, several at 0.32–0.35. Your 15:00 explanation ("full-text Jaccard 0.00") came from this bug, not from the data.

Fix: for every candidate–test pair that matches on *either* signal, compute **both** Jaccards exactly from the two gram sets (cheap at this scale). Then apply the rules. Smaller items:
- `dedup_remove.jsonl` is always written to the same path regardless of `--report`, so a second run overwrites it. Derive it from `--report`.
- `--test` only accepts a directory. Also accept `.jsonl`, because generated data must also be deduplicated against `dev/dev_easy_full.jsonl`.

### B4. 🟡 Low
| Location | Issue | Fix |
|---|---|---|
| [eval_logprob.py:285](src/eval_logprob.py:285) `fertility()` | Uses the first 1,000 rows of a file-sorted dev set, so it covers only about the first 8 subjects. | Use all 1,851, or a seeded stratified sample. Re-report the three numbers. Gemma ≪ Qwen/Llama will still hold, but measure it properly. |
| [eval_logprob.py:185](src/eval_logprob.py:185) calibration | The prior uses `rows[0]`'s subject for all rows, and only one content-free input. | See C-16:24 Q4. |
| [eval_logprob.py:353](src/eval_logprob.py:353) `append_result` | `accs['easy']` raises KeyError on a Medium/Hard dev file. | Write the per-level columns back (`dev_easy`, `dev_medium`, `dev_hard`, `macro`) and leave them empty when missing. You will have a Medium dev soon. |
| [eval_logprob.py:263](src/eval_logprob.py:263) batching | Left padding without explicit `position_ids` means results can differ slightly with batch composition (fp16). | Sort prompts by length before batching. It's also faster. For the final submission, check that batch 1 and batch 8 give identical predictions. |
| `colab_eval.ipynb` cell 2 | `pip install -r requirements-colab.txt` is unpinned, so each Colab session can bring a new `transformers`. | Pin (see Q8). Print versions into each `outputs/*/config.json`. |
| configs `revision: null` | Still unpinned. | Pin the HF commit hash now. The models are already downloaded, so the hash is in the HF cache (`refs/main`). |
| `MANIFEST.md` | "Models: none used yet" is wrong, because three were used. | Add the model, parameter count and revision for each. |
| `colab_eval.ipynb` cells 7 + 11 | Run the same experiments twice (direct + grid). | Keep the grid only. |

### Verified correct
- Permutation mapping (`score_rows`) and its oracle test.
- `digit_variants` (single-token check plus logsumexp over the `"1"` / `" 1"` variants).
- `logits_to_keep` fallback.
- BOS handling for chat vs raw.
- Qwen3 `enable_thinking=False`.
- The Llama fixed date.
- fp32 for Gemma.
- Test runs print no diagnostics.
- `predict.validate` (now checks category, BOM, int `q_no`).
- `norm_model` keeps ZWJ.
- `build_dev` stratifies by file.

---

## C. Answers: `student_comment_2026-10-14_15-00.md`

**Q1. Ignore question-only matches when the options differ?** Agreed in principle: a generic stem such as "choose the correctly spelled word" is not a duplicate. Apply the rule only *after* fixing B3, with the **true** full Jaccard: ignore when q ≥ 0.5 **and** true full < 0.2. Expect your IGNORED count to drop from 80 to about 21, and REVIEW to grow.

**Q2. Embedding check for generated data?** Yes, add it for **generated and collected** data. Use **LaBSE**: it was built for cross-lingual paraphrase and bitext detection and covers Sinhala, so it also catches a test item that was translated or paraphrased. `multilingual-e5-small` is the cheaper fallback. Dedup runs at data-preparation time, not inference, so it **does not count toward the 8B budget** (still list it in MANIFEST). Don't hard-code 0.95. **Calibrate the threshold on your 18 known REMOVE pairs**: choose the highest cosine that still catches all 18, then manually inspect the top 50 pairs just below it. This requires a model download, so it needs the team lead's OK.

**Q3. Keep the 18 overlapping items in Easy dev?** Keep them for now. For zero-shot models the overlap is irrelevant. Add `"overlaps_test": true` to those rows in `dev_easy_full.jsonl`. Once any training data from past papers is used, report **dev_easy_clean (n = 1,833)** as the primary number and the full set as secondary.

**Q4. Keeping newlines in `norm_model`?** Yes. That's the right call for sub-option questions.

**Q5. Stricter threshold for collected past papers?** Yes. Collected exam items have a high prior of recurring:

| Source | Auto-remove | Manual review |
|---|---|---|
| Collected past-paper MCQs | full ≥ **0.35** or LaBSE ≥ calibrated threshold | full 0.25–0.35 |
| Generated MCQs | full ≥ 0.5 or LaBSE ≥ threshold | full 0.35–0.5 |

**Q6. Medium gold dev: which subjects?** One correction first: test Medium has **13 subjects, not ~30** (Hard has 15, and about 43 subject-level pairs exist across all levels). The top 5 subjects make up 69% of Medium. Allocate 120 items proportionally, with a minimum of 4 per subject:

| Subject (Medium test n) | Dev items |
|---|---|
| Buddhism (174) | 19 |
| Civics (159) | 17 |
| Sinhala (159) | 17 |
| Dancing (153) | 16 |
| History (125) | 13 |
| Health (72) | 8 |
| Geography, Catholicism, Communication & Media, Christianity, Agriculture, Design & Construction, Eastern Music (36–41 each) | 4 each = 28 |
| **Total** | **118** |

Collect from **O/L 2018–2024 and grade 9–10 term papers** that have official answer keys. Dedup at the stricter "collected" threshold before an item enters dev.

**Q7. Global-MMLU-si and Belebele in the organiser email?** Yes, add it. Has the A4 email been sent? It isn't mentioned in either comment, and Phase B depends on the answers.

---

## D. Answers: `student_comment_2026-10-14_16-24.md`

**Reading your results:** your interpretation is correct. Every model is inside the chance band, and the 0.6–1B models have no usable Sinhala MCQ signal. That matches the paper, where even 7–8B Llama/Aya/Mistral are at chance. The only durable finding is **tokenizer fertility: Gemma 2.9 vs Qwen 8.8 vs Llama 10.9 tokens/word.** That's a strong prior for choosing Gemma-family models at larger sizes, both for quality and because a Qwen/Llama Sinhala prompt is about 3× longer (slower, and more truncation risk).

**Q1. Finish the 1B grid?** Finish only what is already running, because it's cheap. **Then stop working with 1B models.** Their grid differences can't exceed noise, so they only test the pipeline, and they've done that. Move all GPU time to the 3–4B / 7–8B screening.

**Q2. Decision rule?** Yes, use **paired tests on the same 1,851 items**. Interval overlap is too conservative and ignores the pairing. Rule:
- **Exact McNemar** test on the discordant pairs (binomial test on b vs c). Adopt a change only if **p < 0.05 and the gain is ≥ 1 pp**. Otherwise keep the simpler option.
- Because you test many settings, decide the shortlist of comparisons *before* running them, and don't stack wins found by search. For a final "A vs B" between two strong systems, also report a paired bootstrap 95% CI of the difference.
- Add a small `src/compare.py` that takes two `outputs/<exp>/probs.jsonl` files and prints accuracy A, accuracy B, b, c, the McNemar p-value and the bootstrap CI.

**Q3. 4-bit for 7–8B screening, and run the 7B check first?** Yes to both.
- **NF4 4-bit is fine for screening.** The paper itself used 4-bit NF4 with double quantization, so 4-bit is actually the closer replication.
- The precise sanity target: the paper reports **Qwen2.5-7B-Instruct zero-shot Easy = 30.85%** (Appendix Table 15). With your prompt changes, expect about **28–33%** on the Easy dev. Clearly below that means a pipeline bug.
- Run it **first**, after B1 and B2 are fixed. Use `chat: auto`, `perms: 1`, `calibrate: false`.
- Note for later: an 8B model in fp16 is about 16 GB and won't fit a T4. Your final system will probably run 4-bit or 8-bit, so the final screening should use the same precision as the final submission.

**Q4. Calibration prior: one input or several?** Average **three content-free inputs** ("N/A", empty string, "[MASK]"), as in the original contextual-calibration paper (Zhao et al., 2021). Average their log-probs to get the prior, and use the row's own subject name rather than `rows[0]`'s. It's cheap. But permutation averaging is the more principled debiasing method. For 7–8B models, add **(calibrate = True, perms = 4)** to the grid and compare it with McNemar.

**Q5. Subject ablation on the best 7–8B model?** Yes, one model, full Easy dev, three runs. Do it after the base model is chosen.

**Q6. Pool subjects by category for the report?** Yes. Don't use the `category` field, because it's inconsistent (`Humanities` / `humanities` / `other` for the same subject). Create a fixed `configs/domain_map.yaml` (file → paper domain: Humanities / Social Science / STEM / Language / Business / Other, per paper Table 2). Report domain-level accuracy with CIs in the main text, and per-subject results in the appendix.

**Q7. Spend one Codabench submission on a chance-level system?** **Yes, as soon as B1 and B2 are fixed.** The 3/day limit resets daily, so a pipeline test costs you nothing. It confirms that the zip, keys and string answers are accepted by the real scorer, which removes a deadline-week risk. Use the Gemma-3-1B predictions. Write down beforehand that this is a format test and that its score will not drive any decision.

**Q8. Pin Colab versions instead of Kaggle versions?** Yes. Which platform doesn't matter; what matters is that the versions you **ran with** are the ones you ship. In the notebook, add a cell with `!pip freeze > requirements-lock.txt`, and pin `torch`, `transformers`, `accelerate`, `bitsandbytes` and `peft` exactly in `requirements.txt`. Record the Python version, CUDA version and GPU name in MANIFEST. The organisers rebuild from these files.

---

## E. Next steps

| Order | Task | Done when |
|---|---|---|
| 1 | Fix **B1** (YAML chat), **B2** (truncation), **B3** (dedup). Delete the invalid Qwen rows. | Unit test extended with: a config with `chat: on` resolves to chat ON; a 3,000-token prompt raises an error instead of truncating. |
| 2 | Rerun `dedup.py`, update `dedup_report.txt` + MANIFEST, add `overlaps_test` flags to dev. | New REMOVE / REVIEW / IGNORED counts reported. |
| 3 | Pin requirements + HF revisions, and update MANIFEST models. | `requirements-lock.txt` committed. |
| 4 | **Commit everything** (excluding test data). Nothing is committed yet, and losing this work would be expensive. | `git log` shows a commit. |
| 5 | Smoke test (20 questions) then the **Qwen2.5-7B-Instruct 4-bit sanity check** on the full Easy dev. | Accuracy is about 28–33%, and the max token length is printed. |
| 6 | Codabench format-test submission (Gemma-1B predictions). | Accepted by Codabench. |
| 7 | **Phase A screening, 4-bit, full Easy dev**: Gemma-3-4B-it, Qwen3-4B, Qwen3-8B, Llama-3.1-8B-Instruct, Aya-Expanse-8B, plus any newer ≤ 8B open model released **before 1 Oct 2026** (check the release date on the model card). Settings: chat auto, perms 1. Then perms 4 and calib on the top 2. | Results table with fertility and McNemar vs the best. |
| 8 | Add `src/compare.py` (McNemar + bootstrap). | Used for every decision in step 7. |
| 9 | Start collecting the **Medium gold dev** (C-Q6 table). | ≥ 60 items by the next meeting. |
| 10 | Team lead: send the organiser email (A4 from 13:36, plus your Q7). | Reply saved in `comments/`. |

**Bring to the next meeting:** the fixed `results.csv`, the Qwen2.5-7B sanity number, the Phase-A table (at least 3 models at 3–8B), the new dedup counts, the Codabench format-test outcome, and the status of the organiser email.

— Supervisor
