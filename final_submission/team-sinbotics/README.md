---
license: apache-2.0
language:
- si
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: text-generation
tags:
- sinhala
- multiple-choice
- lora
- sinhala-mmlu
---

# Team SinBotics – Sinhala MMLU Shared Task System

## System summary
- **Base model:** [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (released 2026-02-27, Apache-2.0, open weights).
- **What we did:** We LoRA-fine-tuned Qwen3.5-4B (rank 16, one epoch) on Sinhala multiple-choice questions from Global-MMLU (Sinhala) and our own collection sinmulu_si (25,799 training questions). The loss is on the answer digit only. At inference, the merged model gets a zero-shot prompt: an expert Sri Lankan teacher of the question's subject, asked to think step by step in English, with Qwen's thinking mode on (greedy decoding, up to 4,096 new tokens). The answer is the most probable of the tokens 1/2/3/4 after the reasoning and a final "Answer: " cue, so every question gets exactly one answer.
- **Total inference parameters:** **4.24B**. This is the text model of Qwen3.5-4B (4,205,751,296) plus the LoRA adapter (30,474,240); the adapter is merged into the weights, so the loaded model has 4,205,751,296. The full Qwen3.5-4B checkpoint, including the vision tower we don't load, has 4,659,865,088. Either way, it is ≤ 8B.
- **Extra data used for training:**
  - [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), Sinhala subset `si` (Apache-2.0), revision `0e619dbeb34206cd48705a1a0ea7fb21cae09993`: test and dev splits, 14,324 questions after removing 3 with an empty option.
  - **sinmulu_si**: our own collection of 15,190 Sri Lankan Sinhala multiple-choice questions, gathered by our team; 15,180 after removing duplicates. Before training, it was checked against the 2,238 shared-task test questions, and 165 overlapping questions were removed (see *Overlap check against the test set*). We plan to publish it separately, so it is **not included** in this repo; it is available to the organisers on request.
  - Together: 26,675 training and 2,829 validation questions (10% per subject, seed 42); see `training/README.md`.
  - **How sinmulu_si was digitised:** the exam papers were scanned PDFs in legacy fonts, so their pages were transcribed to text with **Claude Sonnet 5.5** (Anthropic, closed API; used only to create training data, never at inference). Where no official marking scheme was available, the answer was labelled by the model and kept only when it was highly confident (see Datasets).
  - **At inference,** only our open-weight system above is used (≤ 8B parameters), with no external LLM, API or retrieval.

## Models and resources in this submission
| Repo | What it is |
|---|---|
| [`SinhalaMMLU-SharedTask/team-sinbotics`](https://huggingface.co/SinhalaMMLU-SharedTask/team-sinbotics) | this repo: code, inference script, final predictions, training code, and our LoRA adapter on Qwen3.5-4B in [`adapter/`](https://huggingface.co/SinhalaMMLU-SharedTask/team-sinbotics/tree/main/adapter) (rank 16, combined Sinhala data, 1 epoch; weights, config and tokenizer files) |
| [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | base model (downloaded by `inference.py` at the pinned revision) |


## Datasets

### Extra training data (for the LoRA adapter)
| Dataset | Source / licence | Questions | Train / validation |
|---|---|---|---|
| **Global-MMLU, Sinhala** | [`CohereLabs/Global-MMLU`](https://huggingface.co/datasets/CohereLabs/Global-MMLU) (`si`), Apache-2.0 | 14,324 | 12,926 / 1,398 |
| **sinmulu_si** (our own) | Sri Lankan exam papers; not published here (available on request) | 15,180 | 13,749 / 1,431 |
| **Combined** (final adapter) | union of the two | 29,504 | 26,675 / 2,829 |

Validation sets are 10% per subject (seed 42).

**Global-MMLU, Sinhala** (`training/sinmmlu/gmmlu.py`)
- MMLU translated into Sinhala: 57 subjects. Categories: humanities 4,770, STEM 3,245, social science 3,137, other 2,433, business 739.
- Pinned revision `0e619dbeb34206cd48705a1a0ea7fb21cae09993`. We merged the test (14,042) and dev (285) splits and dropped 3 questions with an empty option.
- We removed the metadata, converted the questions to the competition format with ids `gsi00001…`, and split them per subject.

**sinmulu_si** (`training/sinmmlu/sinmulu.py`)
- **Source:** built by our team from 3,486 Sri Lankan exam-paper PDFs: G.C.E. O/L and A/L papers and provincial term tests.
- **Transcription:**
  - Most PDFs use legacy Sinhala fonts, so text extraction was unusable. Instead, each page was rendered as an image and transcribed by **Claude Sonnet 5.5** (Anthropic API).
  - This ran in batches, with several agents working in parallel on separate slices.
  - Stems and options were copied verbatim.
- **Selection:**
  - Only Sinhala-medium papers and true multiple-choice questions were kept.
  - Calculation-heavy subjects (Mathematics, Accounting, Statistics) and English, Tamil and other language papers were excluded.
  - Questions that depended on diagrams, or whose scan was too poor to read confidently, were dropped rather than guessed.
- **Answers:**
  - These come from official marking schemes where available, matched to papers by grade, year and province.
  - Otherwise, an answer was kept only if the model was highly confident, and left blank if not, so some answers are model-derived.
- **Cleaning:** records were validated, de-duplicated and cleaned. That gave 15,355 four-option MCQs across about 48 subjects.
- **Final set:**
  - After the overlap check against the test set (below), 15,190 remain (`osi00001`–`osi15190`). Removing 10 more duplicates (same question and options) leaves 15,180.
  - Largest subjects: Agricultural Science 990, Home Economics 971, Biology 944, ICT 890, Communication and Media Studies 860, History 779.
- **Availability:** we will publish it separately; it is available to the organisers on request.

**Combined** (`training/sinmmlu/combined.py`)
- A validation question whose text also appeared in training would have been moved to training; none did.
- Questions longer than 1,024 tokens were skipped in training, leaving 25,799 / 2,745.
- Training answers are spread evenly: 1: 23.1%, 2: 25.9%, 3: 26.4%, 4: 24.7%.

### Overlap check against the test set (sinmulu_si)
Real exam papers can repeat questions, so before training we checked sinmulu_si (15,355 questions) against all 2,238 shared-task test questions:
1. **Candidates.** We normalised the Sinhala text, then found the closest sinmulu_si candidates for each test question with character n-gram TF-IDF.
2. **Scoring.** Each candidate pair was scored with `difflib` similarity on the question text and on the choices. The choice score is the average of the 3 best-matching choice pairs, ignoring their order.
3. **Manual review.** We reviewed the ranked pairs by hand and decided which were the same question and which only looked similar.
   - **Round 1:** 505 pairs scored ≥ 0.6 on the combined measure. Removing the confirmed duplicates took out **143** questions.
   - **Round 2:** we re-ran the check on the cleaned set at ≥ 0.7 on question, choices or both. Over three passes, a further **22** were removed (6, 7 and 9).
4. **Result.** **15,190** of the 15,355 questions remain (165 removed), renumbered `osi00001`–`osi15190`. Our preparation script then removes 10 exact duplicates within the set, leaving 15,180.

**Remaining similarity, by design:**
- 260 test questions still have a sinmulu_si question with question similarity above 0.7, and 10 of those also match above 0.7 on the choices. We judged most of them to be different template-style questions, for example the same stem pattern about a different item.
- 15 test questions still score 0.99 or higher on question similarity, and these should be double-checked.
- Choice similarity alone (827 test questions above 0.7) is not treated as duplication, because different questions often share the same answer options.

The cleaning therefore relies on manual judgement, not on a similarity cut-off, and it **does not prove zero leakage**. The check used only the test questions and choices, no answers.

All training data use the competition format: `q_no`, `subject`, `question`, `choices` (4 options), `answer` ("1"–"4"). Apart from the Claude Sonnet 5.5 transcription and model-derived answers in sinmulu_si described above (closed API, training data only), we used no synthetic data and no retrieval corpus.

## System in detail

### Pipeline (per question)
1. **Normalise the text.** Unicode NFC, remove zero-width non-joiners (U+200C, an artefact of legacy Sinhala fonts that is much more frequent in the test set than in the training data), keep zero-width joiners (U+200D, needed for conjuncts such as ප්‍ර), and tidy spaces while keeping line breaks so sub-option lists stay readable.
2. **Build the prompt.** It is a single zero-shot user turn with the subject name in English, the original Sinhala question (no translation) and the options numbered 1–4:
   ```text
   You are an expert Sri Lankan teacher of {subject}.
   The following is a multiple choice question in Sinhala about {subject}.
   Think step by step in English, then choose the correct or most suitable answer from the options 1, 2, 3, 4.
   End your reply with "Answer: <number>".

   Question: {question}
   1. {option 1}
   2. {option 2}
   3. {option 3}
   4. {option 4}
   ```
3. **Render the chat template.** We use Qwen3.5's chat template with `enable_thinking=True`, so the model first reasons inside `<think> … </think>`. The template's date is fixed, so the prompt never depends on the run date.
4. **Generate the reasoning.** Greedy decoding (`do_sample=False`), at most 4,096 new tokens, with length-sorted batches of 48 and left padding.
5. **Read off the answer.** If the reasoning ends with "Answer: N", that statement is cut off. The thinking block is closed with `</think>` if needed, and "\n\nAnswer: " is appended. One forward pass then gives the log-probabilities of the tokens `1`, `2`, `3` and `4` at the next position, and the most probable one is the answer. So every question gets exactly one valid answer, including questions whose reasoning hit the 4,096-token limit. In our final run, 97.4% of the stated answers agree with this read-off answer.

### Fine-tuning
| | |
|---|---|
| Method | LoRA (PEFT 0.21.1), rank 16, alpha 32, dropout 0.05 |
| Target modules | q/k/v/o, gate/up/down, and the Gated-DeltaNet projections in_proj_qkv/in_proj_z/out_proj of all 32 text layers; vision tower not used |
| Trainable parameters | 30,474,240 |
| Objective | log P(gold digit) after a plain multiple-choice prompt (`en_instr_nosubject`) whose chat reply is pre-filled with "Answer: ". The loss is on that single token only, so the model is not taught to stop after a digit, and the reasoning prompt above keeps working. |
| Data | 25,799 training questions: Global-MMLU Sinhala + sinmulu_si, 10% per subject held out for validation (seed 42), questions longer than 1,024 tokens skipped. No shared-task test questions were used. |
| Optimiser | AdamW, learning rate 1e-4, cosine schedule with 3% warm-up, weight decay 0, gradient clipping 1.0, 32 questions per step, 1 epoch = 807 steps, seed 42 |
| Precision | bf16 base weights, LoRA weights in fp32 |
| Hardware / time | 1× NVIDIA RTX 3090 24 GB, 8.8 hours |
| Merge | `merge_and_unload()`: the merged model has the same architecture and parameter count as the base text model |

Validation accuracy (2,745 held-out questions, digit prompt) during training:

| Step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 807 |
|---|---|---|---|---|---|---|---|---|---|
| Accuracy (%) | 44.4 | 47.4 | 49.9 | 51.7 | 51.7 | 52.8 | 53.2 | 54.5 | 54.8 |

### How we got here
- **Baseline and prompts.** We first evaluated Gemma-3-4B-it with the prompts of the SinhalaMMLU paper and many variants on development data held out from the public training questions (a screening split and a separate confirmation split). English step-by-step reasoning in the role of a Sri Lankan teacher was the clear winner: 45.6% vs 38.2% for the paper's prompt on the confirmation set.
- **Ideas that did not help enough.** Machine translation of the questions into English (NLLB, MADLAD, self-translation) often lost negations ("which is NOT …"). Few-shot examples, calibration, option-order averaging, self-consistency, a negation-check chain and prompt ensembles each changed accuracy by less than our +2-point threshold.
- **Fine-tuning.** LoRA adapters on Global-MMLU Sinhala and on sinmulu_si improved Gemma-3-4B-it. In all our runs, validation accuracy peaked after one epoch.
- **Final system.** We then moved to Qwen3.5-4B, a newer ≤ 8B open model, trained one LoRA epoch on the combined data, and switched on its thinking mode for the reasoning step.


## Results of our experiments

All numbers are accuracy (%) on our own development data, never on the shared-task test set:
- **Screening / confirmation:** two separate splits of the public training questions; methods were compared on the screening split and the best ones re-tested on the larger confirmation split.
- **Validation sets:** held out from our fine-tuning data.

A condition was adopted only if it beat the baseline by at least 1–2 points with an exact McNemar test (Holm-corrected) on the confirmation set.

### 1. Prompting Gemma-3-4B-it (no fine-tuning)
| Prompt / condition | Screening | Confirmation |
|---|---|---|
| Paper prompt, Sinhala, no subject (baseline A0) | 37.5 | 38.2 |
| Paper prompt + subject (Sinhala) | 42.1 | – |
| Teacher role + subject (Sinhala) | 43.6 | – |
| English instruction | 41.8 | – |
| Few-shot, 3 same-subject examples | 42.5 | – |
| Calibration | 34.3 | – |
| Cloze scoring of option texts | 31.4 | – |
| Reasoning in Sinhala (`cot_si`) | 43.2 | – |
| Reasoning in English (`cot_en`) | 47.9 | 44.9 |
| English reasoning + 2 worked examples | 50.7 | 44.8 |
| Ensemble of English + Sinhala reasoning | 49.3 | 44.0 |
| **English reasoning, teacher role (`cot_en_role`)**, 512 tokens | 48.2 | **45.6** |
| `cot_en_role`, 1,024 tokens | 48.9 | 45.6 |
| Ensemble of three English-reasoning prompts | 50.4 | 46.7 |

Gemma-3-1B-it scored 25.7% on the screening set, about chance.

### 2. Ideas tested on top of `cot_en_role` (screening set; reference 48.9%)
| Idea | Accuracy | Change |
|---|---|---|
| Option-order averaging (2 orders) | 49.6 | +0.7 |
| Self-consistency (5 samples, T = 0.7) | 47.5 | −1.4 |
| Negation check before reasoning | 51.4 | +2.5 (not confirmed: 44.3 on the confirmation set) |
| Judge each option separately | 38.9 | −10.0 |
| Negation-consistency chain (3 turns) | 47.5 | −1.4 |
| Bilingual input, MADLAD translation | 49.3 | +0.4 |
| Bilingual input, NLLB translation | 47.9 | −1.1 |
| English-only input (MADLAD translation) | 44.3 | −4.6 |
| Translate first, then reason (one prompt) | 50.4 | +1.4 |

**Negation:** translations kept the negation in only 56–62% of negated questions. On those questions, Sinhala-input reasoning scored 58.8%, while English-only translated input scored 29.4%. So we never translate.

### 3. LoRA fine-tuning (validation accuracy, digit prompt)
| Model / adapter | Training data | Validation set | Before → after |
|---|---|---|---|
| Gemma-3-4B-it, rank 16, epoch 1 | Global-MMLU si | Global-MMLU dev part | 40.1 → 41.5 |
| Gemma-3-4B-it, rank 16, epoch 1 | sinmulu_si | 5% of sinmulu_si (714) | 40.9 → 47.1 |
| Gemma-3-4B-it, rank 16, epoch 2 | sinmulu_si | same | 47.1 → 47.3 |
| Gemma-3-4B-it, rank 16, epoch 3 | sinmulu_si | same | 47.3 → 45.8 (overfitting) |
| Gemma-3-4B-it, rank 8, epoch 1 | sinmulu_si | 10% of sinmulu_si (1,431) | 37.5 → 43.8 |
| Gemma-3-4B-it, rank 8, epoch 2 | sinmulu_si | same | 43.8 → 44.4 |
| **Qwen3.5-4B, rank 16, epoch 1 (final)** | Global-MMLU si + sinmulu_si | 10% of both (2,745) | **44.4 → 54.8** |

With `cot_en_role` reasoning on the screening set, the Gemma sinmulu_si adapter scored 46.8% after epoch 1 and 51.4% after epoch 2; the base model scored 47.1% in the same run.

## How to run
```bash
pip install -r requirements.txt
python inference.py
```
- **GPU used:** 1× NVIDIA A100-SXM4 40 GB. Generation batches of 48 need about 24 GB of GPU memory; a batch that runs out of memory is split automatically.
- **Approximate run time on the test set:** about 5 hours for 2,238 questions on the A100. Most of it is the thinking-mode reasoning.
- **Anything else the organizers need to know:**
  - No login is needed: the base model, the adapter and the code are public, and nothing is gated.
  - Our run also had `causal-conv1d==1.7.0` installed. It can't be installed from `requirements.txt` (it needs torch at build time). For identical kernels, run `pip install causal-conv1d==1.7.0 --no-build-isolation` after the requirements; without it, transformers uses its torch implementation.
  - Decoding is greedy and fully deterministic on a given GPU. Different GPUs, or different batch compositions, can change a small number of answers through bf16 rounding.
  - The final file was produced by our package (`training/sinmmlu`, `python -m sinmmlu.evaluate --model configs/models/qwen35_4b_lora_combined.yaml --data test --prompt cot_en_role_think --chat on --max-new-tokens 4096`). `inference.py` is a self-contained copy of that code path. We checked that it builds the same prompt for all 2,238 test questions and processes the reasoning texts the same way.
  - On our side, part of the questions was generated in a separate run and reused, so batches were composed differently. A from-scratch run can therefore differ from `predictions.jsonl` in a few answers.

## Final run
| | |
|---|---|
| Questions | 2,238 (easy 642, medium 1,111, hard 485) |
| Hardware | 1× NVIDIA A100-SXM4 40 GB, CUDA 12.8, Python 3.12, torch 2.8.0, transformers 5.18.0, peft 0.21.1, flash-linear-attention 0.4.0, causal-conv1d 1.7.0 |
| Settings | bf16, generation batch 48, scoring batch 16, greedy, at most 4,096 new tokens, seed 42 |
| Answer stated in the reasoning | 56.0% of questions; the rest hit the 4,096-token limit and are read off after the cue |
| Probability on the four answer tokens | 99.9% on average |
| Longest input (prompt + reasoning) | 4,865 tokens |
| Format check | 2,238 lines, one per question, answers "1"–"4", fields `q_no`, `subject`, `category`, `level`, `answer` |

The full configuration of this run is in `training/logs/final_run_config.json`.

## Limitations
- **Long reasoning.** About 44% of the reasoning texts reach the 4,096-token limit before stating an answer. For those questions, the answer is read off from the model's next-token probabilities after a truncated reasoning, which is less reliable.
- **Tokenisation.** Sinhala costs many tokens in the Qwen3.5 tokenizer (about 4.8 tokens per word), which makes long questions and the reasoning about them expensive.
- **Speed.** Thinking mode makes inference slow: about 5 hours for 2,238 questions on an A100.
- **Reproducibility.** Results are deterministic on the same GPU and batch composition. Other GPUs can change a few answers through bf16 rounding.

## Files
- `inference.py` – the exact system that produced our final Codabench submission (self-contained; see above)
- `predictions.jsonl` – the exact file we submitted to the final leaderboard
- `adapter/` – our LoRA adapter (weights, config, tokenizer files, training config and log; `adapter/README.md`)
- `requirements.txt` – package versions we used
- `training/` – data preparation, LoRA training, merging and evaluation code (`sinmmlu` package), configs, the training script and our final run's configuration (`training/README.md`)

## Citation
This work uses the SinhalaMMLU dataset:

> Ashmari Pramodya, Nirasha Nelki, Heshan Shalinda, Chamila Liyanage, Yusuke Sakai, Randil Pushpananda, Ruvan Weerasinghe, Hidetaka Kamigaito, and Taro Watanabe. 2025. SinhalaMMLU: A Comprehensive Benchmark for Evaluating Multitask Language Understanding in Sinhala. In *Proceedings of EMNLP 2025*.

Training data: Shivalika Singh et al. 2025. Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation. In *Proceedings of ACL 2025*.

Base model: Qwen Team, [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0).

## Team
- **Team name:** SinBotics

**Members:**

| Name | Affiliation | E-mail |
|---|---|---|
| Sachintha Weththasinghe | SLIIT | akalankasachintha125@gmail.com |
| Hasitha Hirushan | SLIIT | hasithahja@gmail.com |
| Hasaan Saadhiq | SLIIT | hsaadhiq@gmail.com |
| Janith Ramanayake | UoM | janithramanayake523@gmail.com |
