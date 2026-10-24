# Lab notebook (newest first)

## 2026-10-24 night: confirmation of the 1024-token version; submission 2
inference_rest (vs cot_en_role at 512 tokens, 45.6%): cot_en_role at 1024 tokens 45.6% (+0.0, identical accuracy);
cot_en_role_negcheck at 1024 44.3% (-1.3, p 0.33). Neither adopted; by the agreed fallback, submission 2 =
cot_en_role at 1024 tokens. Test run: 4,282 s, longest input 1,371 tokens, stated-answer rate 99.1% (62.4% at 512),
format validated on the pod and locally; 140 of 2,238 answers differ from submission 1.
File: submissions/2026-10-24_cot_en_role_mt1024/predictions.zip

## 2026-10-24: question-type handling on the subset (reference cot_en_role at 1024 tokens: 48.9%)
- cot_en_role_negcheck ("first state whether the question asks for the correct or the NOT-correct option"):
  **51.4% (+2.5, p 0.48)** -> meets the pre-stated +2 rule, goes to confirmation. But on the 34 negation questions it
  scores 50.0% vs 58.8% for the reference; the gain comes from non-negation questions (52.5% vs 47.5%), so it is
  probably wording noise rather than better negation handling.
- cot_en_role_each_option ("judge each option correct/incorrect"): 38.9% (-10.0, p 0.001): clearly harmful.
- Question types (subset: negation 34, sub-option 5, blank 16, numeric 30, plain 202). With English reasoning,
  negation questions are NOT a weak spot (58.8% vs 47.5% plain); they are weak only for direct scoring (35.3%) and for
  translated input (29-38%). Fill-in-the-blank 62.5%, numeric 43.3%; sub-option n=5 is too small to judge.

## 2026-10-24: translation + reasoning on the subset (reference cot_en_role at 1024 tokens: 48.9%)
Bilingual input (Sinhala + English translation): MADLAD 49.3% (+0.4), NLLB 47.9% (-1.1), self 41.8% (-7.1, p 0.04).
English-only (MADLAD) 44.3% (-4.6). Translate-first in one prompt (no extra model) 50.4% (+1.4, p 0.72). Ensemble
Sinhala-only + bilingual MADLAD 50.4% (+1.4). **No condition reaches the pre-stated +2 points: none goes to
confirmation; the single-model system stays.**
Negation: 34/280 subset questions contain a Sinhala negation word. Translations keep an English negation in only
62% (MADLAD), 59% (NLLB), 56% (self) of them. On these 34 questions, Sinhala-only reasoning scores 58.8% but bilingual
MADLAD 38.2% and English-only MADLAD 29.4%: lost negations are the clearest way translation hurts (n=34, indicative).
Side-by-side file: outputs/analysis/translations_inference_subset.md on the pod.

## 2026-10-24: cheap ideas (committed as f11ed8e before starting; pushed to origin/initial-setup)
- New: `--max-new-tokens` (reasoning limit) and `--samples k --temperature t` (self-consistency: k sampled reasoning
  texts, seeds 42+1000*i, top-p 0.95, scored label distributions averaged; hard majority vote reported as
  majority_vote_acc). Experiment ids get `__mt<N>` / `__sc<k>t<t>` only when set.
- Ensemble of the three confirmed English-reasoning prompts (cot_en + cot_en_role + cot_en_fewshot2):
  subset 50.4%; **inference_rest 46.7%** (+8.5 vs A0; vs cot_en_role +1.1, 95% CI [-0.9, +3.0], p 0.31).
  Note: chosen after the confirmation results were known, so its rest number is optimistic.
- Reasoning limit, cot_en_role on the subset: 512 tokens 48.2% (parse 79.3%); **768 / 1024 / 1536 / 2048 tokens all
  48.9%** with identical predictions (parse 98.9% / 100% / 100% / 100%): Gemma finishes every Easy answer within
  ~768 tokens, so the limit only matters for answers that were cut at 512. cot_en at 1024: 47.9% (= 512; parse 99.6%).
- Self-consistency, cot_en_role, 5 samples, T=0.7, 1024 tokens: 47.5% soft vote and 47.5% majority vote, i.e. no
  better than greedy (48.9%) at 5x the cost. Not pursued.

## 2026-10-24: correction: parameter counts
count_params summed meta-device parameters without tying, so tied embeddings were counted twice. Counted from the
stored safetensors (runpod/count_params.py): Gemma-3-4B-it/pt **4,300,079,472** (vision tower 419,816,304), MADLAD-400-3B
**2,940,374,016** (its lm_head is stored untied), NLLB-200-distilled-600M **615,073,792** (unique tensors; its .bin stores
the tied embedding several times: 1,402,138,624 stored values). **Gemma + MADLAD = 7.24B <= 8B: the earlier statement that
MADLAD is over the limit was wrong.** It does not change the selection (MADLAD-en 43.9% on the subset, below cot_en, and
not a confirmation candidate under the rule). count_params now reads the safetensors headers.

## 2026-10-24: first test predictions (submission 1, not uploaded yet)
cot_en_role (Gemma-3-4B-it, bf16, English reasoning with a teacher role, greedy, max 512 new tokens) on all 2,238
public test questions: 3,083 s on the RTX 3090; format validated on the pod and again locally.
File: submissions/2026-10-24_cot_en_role/predictions.zip (+ predictions.jsonl, config.json). System parameters:
4,971,331,952 (one model). Process note: while checking the file I printed the predicted-answer counts on test; nothing
was chosen from them, and test-set diagnostics stay off from here on.

## 2026-10-24: confirmation on dev/inference_rest (n=1,483, ±2.5 pp)
A0 38.2%. All four candidates adopted (gain >= 1 pt, Holm p < 0.05): **cot_en_role 45.6% (+7.4, Holm 1.2e-6)**,
cot_en 44.9% (+6.7), cot_en_fewshot2 44.8% (+6.6), ens(cot_en, cot_si) 44.0% (+5.9). Winner by the rule: cot_en_role.
The subset winner (cot_en_fewshot2, 50.7% on 280) regressed to 44.8%: the 280-question ranking among the reasoning
prompts was noise; the three English-reasoning variants are within ~1 pt of each other. Parse rates: cot_en 97.0%,
cot_en_fewshot2 95.5%, cot_en_role 76.6% (scored anyway, after the answer cue), cot_si 97.4%.
Test predictions (2,238) started with cot_en_role.

## 2026-10-24: combinations on dev/inference_subset (vs A0 37.5%)
Rationales: Gemma's English reasoning was correct on 28/70 pool questions (none for Buddhism -> generic fallback).
**cot_en_fewshot2 50.7% (+13.2, Holm 0.003)**; ens(cot_en, cot_si) 49.3% (+11.8, Holm 0.005); cot_en_role 48.2%
(parse rate only 79%); cot_en 47.9%; pt fewshot3 47.1%; cot_en_subject 46.8%; pt fewshot3_subject_si 46.8%;
ens(cot_en, role, fewshot3) 44.6%; ens(4 scored prompts) 44.3%; it fewshot3_subject_si 42.9%.
Confirmation candidates by the rule above: cot_en_fewshot2, ens_cot_si_en, cot_en_role, cot_en (+ A0).

## 2026-10-24: confirmation rule (written before the combination results were seen)
Candidates for dev/inference_rest.jsonl (n=1,483): the 4 best conditions on the subset after the combination runs,
counting only valid ones (label mass >= 0.5; total parameters <= 8B, so no MADLAD; ensembles allowed when all parts
use the same model), plus the baseline A0. A condition is adopted only if, on inference_rest, it beats A0 by >= 1 point
with Holm-corrected exact McNemar p < 0.05 over the confirmed comparisons; among adopted ones, the highest accuracy wins.

## 2026-10-24: wave 3 on dev/inference_subset (vs A0 37.5%)
- **cot_en 47.9% (+10.4, p 0.0013, Holm 0.03)**: the only condition significant after Holm. Parse rate 96.4%;
  350 s for 280 questions. cot_si 43.2% (+5.7, p 0.085), parse rate 97.9%, 450 s.
- MADLAD: first run out of memory (beam 4, batch 32, 24 GB); fixed with batch halving. en_nosubject 43.9% (+6.4),
  bilingual 42.9%, Sinhala q + English options 37.9%. **MADLAD-400-3B has 3,464,662,016 parameters: with Gemma-3-4B-it
  (4,971,331,952) that is 8.44B > 8B, so MADLAD cannot be part of a submission.**

## 2026-10-24: wave 2 on dev/inference_subset (vs A0 37.5%)
Few-shot (fixed same-subject examples): 1-shot +3.2, **3-shot +5.0** (p 0.12), 5-shot +4.6. Translation does not help:
en_nosubject NLLB -0.4 / self +0.4; bilingual NLLB +2.9 / self 0.0; Sinhala question + English options NLLB -1.4 /
self -2.5. Label mass >= 0.997 everywhere. NLLB translated 1,402 sentence segments in ~1 min.

## 2026-10-24: wave 1 on dev/inference_subset (n=280, chance 27.5%, ±5.6 pp)
Baseline A0 (paper_nosubject, chat, 1 order, no calibration) **37.5%**. Gains vs A0 (exact McNemar p / Holm p):
pt base model raw +7.5 (0.046 / 0.55); role_subject_si +6.1 (0.030 / 0.39); paper_subject_si +4.6 (0.09);
en_instr +4.3 (0.09); subject_en +3.2; perms4 +1.1; perms4+calib +0.7; chat off 0; 4-bit -0.7; strict -0.7;
letters -1.1 (invalid, label mass 0.002); calibration -3.2; cloze -6.1; Gemma-3-1B -11.8 (25.7%, Holm 0.045).
Nothing survives Holm at n=280; promoted to confirmation: pt base, role_subject_si, paper_subject_si, en_instr.
4-bit costs nothing measurable (useful if the organisers reproduce on a small GPU). Runs take 7-30 s each.

## 2026-10-24: RunPod RTX 3090; answer pre-fill fix
- Pod: RTX 3090 24 GB, torch 2.8.0+cu128, transformers 5.18.0, bitsandbytes 0.50.2. Model cache on the container
  disk (/root/hf_cache): the template presets HF_HOME on the 30 GB volume, which overflowed during the MADLAD download.
- Gemma-3-4B-it revision 093f9f388b31de276ce2de164bdc2081324b9767, **4,971,331,952 parameters** incl. vision tower.
- Smoke test found: zero-shot label mass ~0.05. Gemma starts its reply with "වැඩිම ගැළපෙන පිළිතුර: 2" or "**", and
  encodes " 2" as a space token + "2". Probe on 40 subset questions (label mass): chat no pre-fill 0.053;
  pre-fill "පිළිතුර:" 0.000; **pre-fill "පිළිතුර: " 1.000**; raw no space 0.001, raw + " " 0.999; English
  "Answer: " 1.000. Fix: every label-scored prompt is pre-filled with "<answer word>: " (chat) or " " (raw); few-shot
  answers are written the same way; reasoning cues end with ": ". The students' phase-0 "chat off" runs had the same
  problem (mass 0.000).
- Letter labels (B6): label mass ~0 even with the pre-fill; Gemma answers with a Sinhala letter (ඊ) or a digit.

## 2026-10-24: screening code for RunPod (A40), all three waves approved
- New: few-shot pool (`dev/fewshot_pool.jsonl`, 5 per subject, first four cover all four answers; taken out of
  `inference_rest`, now 1,483), prompts B4-B7 / cloze / reasoning / translated (prompts.py), letter labels,
  `LM` wrapper (label scoring, cloze, batched greedy generation), `translate.py` (NLLB, MADLAD, self), `ensemble.py`,
  `runpod/` (setup, download, run_subset.sh smoke|wave1|wave2|wave3|report, pack).
- Reasoning runs: greedy generation, then the stated answer is cut and the label probabilities are scored after the
  answer cue, so every question gets probabilities; parse rate and agreement with the stated answer are recorded.
- 29 offline tests pass, including a fake bigram model that checks label and cloze scoring against an unbatched
  reference. Not yet run on a real model (no GPU stack locally): the smoke wave is the first real test.

## 2026-10-24: inference subset
`python -m sinmmlu.build_dev` now also writes `dev/inference_subset.jsonl`: 20 random questions from each of the 14
subjects (280, seed 42), drawn from `easy_clean` (no test near-duplicates). `dev/inference_rest.jsonl` (1,553) is the
disjoint confirmation set. uids: `configs/subsets/inference_subset_uids.txt`. Gold answers 1/2/3/4 = 77/74/74/55, so
always answering "1" scores 27.5% (the chance line for this subset); 95% CI at n=280 is about ±5.4 points.

## 2026-10-24: codebase cleanup (single owner from now on)
- Replaced the student `src/` scripts with the `sinmmlu/` package; old code, notebooks, cluster package and the
  phase-0 results are in `archive/`. A full backup of the pre-cleanup tree was taken before moving anything.
- Verified: same data counts (train 1,851, test 642/1,111/485), same dedup result (18/100/21), and the new scoring
  gives identical probabilities to the old code on all 16 prompt × perms × calibration settings, except
  English-subject + calibration, where the old code built the calibration prompts with the Sinhala subject name (bug).
- Fixed on the way:
  - 4-bit used bitsandbytes' default FP4 with fp16 compute; now NF4 (as in the paper) with bf16/fp32 compute.
  - The grid forced full precision; `--quant` now works.
  - No check for NaN/inf logits; now raises.
  - Runs now record total parameter count, bitsandbytes/peft versions and the git revision.
  - Subject spelling variants inside one file no longer matter: the subject key is the source file.
- Not available locally: transformers/peft are not installed and torch is the CPU build; D: has ~8 GB free
  (Gemma-3-4B is ~8.6 GB), so the HF cache must live on C: or a GPU runtime must be used.

## 2026-10-15: phase 0 (students)
Qwen3-0.6B, Gemma-3-1B-it, Llama-3.2-1B-Instruct on all 1,851 Easy dev questions: all at chance (22–28%).
Digit mass with the chat template: Gemma 0.98, Llama 0.87, Qwen 0.03. Fertility (tokens per word): Gemma 2.96,
Qwen 8.66, Llama 10.74. Details: archive/phase0_1b/results/results.csv.
