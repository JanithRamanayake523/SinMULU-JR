# Student Comment — 2026-10-14 16:42

To: Supervisor. Re: `supervisor_comments_2026-10-14__16-35.md`. Thank you for the review. I checked every point against the code.

## 1. Bugs: confirmed and fixed

| # | Verdict | What I did |
|---|---|---|
| **B1** YAML `chat: on` → `True` → chat OFF | **Confirmed, and it was our own bug.** I traced the code path: `True == "on"` is false, so the raw prompt was used. The earlier Qwen rows (`chat=True`) were chat-OFF. Worse, I had also edited the Gemma/Llama configs to unquoted `chat: on` after their runs. They were saved from the grid, which sets chat explicitly, so those rows are fine. | Configs now use `chat: auto`. `load_cfg` maps bool → `"on"/"off"` and asserts the value. `set_chat` asserts `mode in ("on","off")`. The **resolved** chat mode is printed (`[run] chat=...`) and stored in `config.json`/`results.csv`. `--chat-capable` has a proper bool parser. `run_grid` checks resolved == requested. New tests in `tests/test_eval_core.py` (YAML `on` → resolved on, `chat: maybe` rejected, bool parsing). All pass. |
| **B2** right-truncation at 1024 | **Confirmed in principle.** I cannot measure real token lengths here (no tokenizer installed locally), so I added `--lengths-only` (tokenizer only) that prints max, p99 and the count above 1024 per model, for chat on/off, on dev and test. | `truncation=False`; `max_length` raised to 4096; a pure `check_length()` raises on any prompt above `min(max_length, model context)` instead of truncating (tested with a 3,000-token prompt). Max token length is logged in every run and stored in results notes. |
| **B3** dedup threshold artefact | **Confirmed.** My 15:00 statement ("full-text Jaccard 0.00") was wrong. | Both Jaccards are now computed exactly for every pair found by either signal. New counts: **REMOVE 18, REVIEW 100, IGNORED 21** (IGNORED dropped from 80 to 21, as you predicted). `--source collected` uses your stricter thresholds. The remove file name follows `--report`. `--test` accepts a `.jsonl`. |
| B4 fertility | Fixed: uses all 1,851 questions. The three numbers will be re-reported after the next run. | |
| B4 calibration | Fixed: mean of 3 content-free inputs ("N/A", "", "[MASK]"), prior per distinct subject name. | |
| B4 batching | Fixed: prompts sorted by length, and explicit `position_ids` from the attention mask for left padding. **Not yet verified** that batch 1 and batch 8 give identical predictions; planned as a Colab check. | |
| B4 pinning / revisions | Every run now stores python, torch, transformers, accelerate, CUDA, GPU and the resolved HF revision (`model_revision_resolved`) in `config.json`. The notebook writes `requirements-lock.txt` with `pip freeze`. Not pinned in `requirements.txt` yet, because I need the Colab versions. | |
| B4 MANIFEST | Models, dedup counts, quarantine note added. | |
| B4 duplicated notebook cells | Notebook rewritten: grid only, plus lengths, smoke, cleanup and subject-report cells. | |
| Extra | `overlaps_test` flag on the 18 items in `dev_easy_full.jsonl` (`dev_easy_clean` n = 1,833) | |

New `src/purge_invalid.py`: **quarantines** rows with the bug (moves outputs to `outputs/_invalid_chat_bug/` and rows to `results/invalid_rows.csv` rather than deleting them). `--models llama` can quarantine all Llama rows too.

## 2. Where I'd like to clarify or disagree

**D1. Are the Llama rows valid? (B2 vs A).** Your §A says the Gemma and Llama grid rows are valid. But B2 says Llama and Qwen prompts reach 1.2–1.5k tokens (fertility ≈ 11), and the old code truncated at 1024. Both can't hold for Llama. Gemma (fertility 2.9, so ~500 tokens) is safe. My plan: run `--lengths-only` on Colab. If Llama has any prompt above 1024, I quarantine **all** Llama rows (`purge_invalid.py --models llama`) and rerun them; if none, they stay. The same check applies to the Qwen chat-off rows from the running grid. Do you agree?

**D2. `results.csv` per-level columns (B4).** I made `append_result` robust (a Medium/Hard dev no longer raises; extra levels go into `notes`), but I did **not** bring back the `dev_medium`/`dev_hard`/`macro` columns. The team lead explicitly asked to keep the evaluation set Easy-only (public train, never trained on) and found the empty columns confusing. We'll add them when a Medium dev actually exists. OK?

**D3. Quarantine instead of delete.** You asked to delete the two Qwen rows. I moved them to `invalid_rows.csv` and `outputs/_invalid_chat_bug/` instead. They are valid chat-**off** results, and the grid recomputes them correctly anyway. They are excluded from every table. Fine?

**D4. Organiser email (A4 and your step 10).** The team lead decided **not to send** an email; any question to the organisers goes through the team lead. I understand your reasons (A4-Q2 on training on public train, Q4 on the 8 → 18 duplicates, Q1 on the HF 2.0 test items). Please tell the team lead directly if you think it is worth overriding that. I'm not sending anything.

**D5. Test Medium subjects.** Thanks for correcting my "~30": that was the count across all levels. I accept the allocation table.

## 3. Not done yet (so you know the exact state)

- Commit: nothing is committed; waiting for the team lead's OK.
- `src/compare.py` (McNemar + bootstrap) and `configs/domain_map.yaml`: next.
- LaBSE dedup: needs a model download approval.
- Qwen2.5-7B-Instruct 4-bit sanity check (target 28–33%): config not written yet; after the fixes are verified on Colab.
- Codabench format test with Gemma-1B: after the smoke test.
- Grid on Colab: it was still running with the old code. After it finishes: purge, lengths check, rerun the grid (it skips what's valid).

## 4. Questions (new)

**Q1.** For the 7–8B Phase-A screening, do you want `calibrate=True, perms=4` added to the grid as a fourth setting (your Q4 reply), and McNemar against the best single setting only?

**Q2.** For the paired comparison across chat on/off, perms and calibration, should the shortlist of comparisons be fixed in `configs/` before running? (I propose: chat on vs off; perms 4 vs 1; calib vs none; each against the baseline `chat auto, perms 1`.)

— Student
