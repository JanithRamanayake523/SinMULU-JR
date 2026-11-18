"""Fill the SinhalaMMLU 2026 system-report template for Team SinBotics.
python build_report.py <template.docx> <out.docx>
Text inside {{...}} becomes a yellow-highlighted placeholder for the team to fill in."""
import copy
import re
import sys

import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt, RGBColor

src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src)
P = list(d.paragraphs)
BODY = Pt(10.5)
CELL = Pt(9)


def runs(p, text, size=BODY, bold_all=False, italic=False, font=None):
    """Append text to p; **bold** spans and {{placeholders}}."""
    for part in re.split(r"(\*\*.+?\*\*|\{\{.+?\}\})", text):
        if not part:
            continue
        r = p.add_run()
        if part.startswith("**"):
            r.text, r.bold = part[2:-2], True
        elif part.startswith("{{"):
            r.text, r.bold = ("[TO FILL: " + part[2:-2].strip() + "]") if part[2:-2].strip() else "[TO FILL]", True
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            r.text = part
            r.bold = bold_all or None
        r.italic = italic or None
        r.font.size = size
        r.font.color.rgb = RGBColor(0, 0, 0)
        if font:
            r.font.name = font
            r._element.rPr.rFonts.set(docx.oxml.ns.qn("w:eastAsia"), font)
            r._element.rPr.rFonts.set(docx.oxml.ns.qn("w:cs"), "Nirmala UI")  # Sinhala glyphs


def clear(p):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    for h in p._p.findall(docx.oxml.ns.qn("w:hyperlink")):
        p._p.remove(h)


def fill(p, label, text):
    clear(p)
    if label:
        runs(p, label + " ", bold_all=True)
    runs(p, text)


def after(p, text, label=None, size=BODY, font=None, style_from=None):
    """New paragraph after p (same paragraph properties), returns it."""
    new = copy.deepcopy((style_from or p)._p)
    for child in list(new):
        if child.tag != docx.oxml.ns.qn("w:pPr"):
            new.remove(child)
    p._p.addnext(new)
    q = docx.text.paragraph.Paragraph(new, p._parent)
    if font:  # code line: no bullet, small indent, no spacing
        ppr = new.find(docx.oxml.ns.qn("w:pPr"))
        if ppr is not None:
            for np_ in ppr.findall(docx.oxml.ns.qn("w:numPr")):
                ppr.remove(np_)
        q.paragraph_format.left_indent = Pt(18)
        q.paragraph_format.first_line_indent = Pt(0)
        q.paragraph_format.space_before = Pt(0)
        q.paragraph_format.space_after = Pt(0)
    if label:
        runs(q, label + " ", bold_all=True, size=size)
    runs(q, text, size=size, font=font)
    return q


def chain(p, items, **kw):
    for it in items:
        if isinstance(it, tuple):
            p = after(p, it[1], label=it[0], **kw)
        else:
            p = after(p, it, **kw)
    return p


def set_cell(c, text, bold=False, size=CELL):
    for para in c.paragraphs[1:]:
        para._p.getparent().remove(para._p)
    p = c.paragraphs[0]
    clear(p)
    runs(p, text, size=size, bold_all=bold)


def fill_table(t, rows, header=None):
    """rows: list of lists; the template's first data row is the format for all rows."""
    proto = copy.deepcopy(t.rows[1]._tr)
    for r in list(t.rows)[1:]:
        r._tr.getparent().remove(r._tr)
    if header:
        for c, h in zip(t.rows[0].cells, header):
            set_cell(c, h, bold=True)
    for vals in rows:
        tr = copy.deepcopy(proto)
        t._tbl.append(tr)
        row = t.rows[-1]
        for c, v in zip(row.cells, vals):
            bold = isinstance(v, str) and v.startswith("!")
            set_cell(c, v[1:] if bold else v, bold=bold)


def table_after(p, proto_table, header, rows):
    new = copy.deepcopy(proto_table._tbl)
    p._p.addnext(new)
    t = docx.table.Table(new, p._parent)
    # match the column count of the prototype: header length must equal it
    fill_table(t, rows, header=header)
    spacer = copy.deepcopy(p._p)
    for child in list(spacer):
        if child.tag != docx.oxml.ns.qn("w:pPr"):
            spacer.remove(child)
    new.addnext(spacer)
    return docx.text.paragraph.Paragraph(spacer, p._parent)


# ------------------------------------------------------------------ header + remove "How to submit"
fill(P[1], None, "System Description Report")
for p in P[2:8] + [P[17], P[21], P[33], P[44], P[48], P[57]]:
    p._p.getparent().remove(p._p)

# ------------------------------------------------------------------ title and team table
fill(P[9], "Title:", "SinBotics at Sinhala MMLU 2026: LoRA-Tuned Qwen3.5-4B Thinking in English about Sinhala "
     "Questions")
T = d.tables
fill_table(T[0], [
    ["Team name", "SinBotics"],
    ["Codabench username", "sachintha_akalanka"],
    ["Members and affiliations", "Sachintha Weththasinghe, SLIIT\nHasitha Hirushan, SLIIT\nHasaan Saadhiq, SLIIT\n"
                                 "Janith Ramanayake, University of Moratuwa"],
    ["Contact email", "akalankasachintha125@gmail.com (Sachintha Weththasinghe)"],
    ["Hugging Face repositories", "SinhalaMMLU-SharedTask/team-sinbotics (code, inference script, predictions, and "
                                  "our LoRA adapter in adapter/)"],
    ["Final leaderboard scores (%)", "Easy 54.67   Medium 52.30   Hard 65.77   Macro Avg 57.58"],
])
# multi-line members cell: split into paragraphs
cell = T[0].rows[3].cells[1]
lines = cell.paragraphs[0].text.split("\n")
set_cell(cell, lines[0])
prev = cell.paragraphs[0]
for ln in lines[1:]:
    prev = after(prev, ln, size=CELL)

# ------------------------------------------------------------------ 1. abstract
fill(P[11], None,
     "We fine-tune Qwen3.5-4B with a rank-16 LoRA adapter on 25,799 Sinhala multiple-choice questions: Global-MMLU "
     "Sinhala and sinmulu_si, our own collection of Sri Lankan exam questions, which we transcribed with Claude "
     "Sonnet 5.5 and cleaned of test overlap. At inference the model thinks step by step in English about the "
     "untranslated Sinhala question, in the role of a Sri Lankan teacher of the subject, with Qwen's thinking mode "
     "on. The answer is the most probable digit after a final \"Answer:\" cue. The system has 4.24B parameters. In an "
     "earlier phase with Gemma-3-4B-it, English reasoning over the Sinhala question was the largest single gain "
     "(38.2% to 45.6% on held-out public training questions); machine translation hurt, mainly by losing negations. "
     "Final leaderboard Macro Avg: 57.58%.")

# ------------------------------------------------------------------ 2. system overview
fill(P[13], None,
     "A test question (subject, Sinhala question, four options) is NFC-normalised; zero-width non-joiners, a "
     "legacy-font artefact, are removed, and zero-width joiners, needed by Sinhala conjuncts, are kept. The question "
     "is not translated. **(1) Prompt:** one zero-shot user turn of Qwen's chat template with thinking mode on; the "
     "model is an \"expert Sri Lankan teacher of {subject}\", asked to think step by step in English and to end with "
     "\"Answer: <number>\" (Section 5.2). **(2) Reasoning:** greedy decoding of at most 4,096 new tokens with the "
     "merged model (Qwen3.5-4B + our LoRA adapter). **(3) Answer:** any stated answer is cut off, the thinking block "
     "is closed if needed, and \"Answer: \" is appended. One forward pass gives the probabilities of the tokens "
     "1–4, and the most probable is the answer, so every question gets exactly one valid answer, even when the "
     "reasoning was cut off.")
fill(P[14], "Approach type:", "LoRA adapter (fine-tuning) + prompting (English chain-of-thought in Qwen's thinking "
     "mode). No ensemble, no retrieval.")
fill(P[15], "Why you chose this approach:",
     "Without training, English reasoning over the original Sinhala was the most reliable gain in our experiments, "
     "and a small LoRA adapter adds Sinhala exam knowledge cheaply within the 8B limit. Qwen3.5-4B was trained for "
     "long reasoning, and its thinking mode helped most on Hard questions.")

# ------------------------------------------------------------------ 3. models
fill_table(T[1], [
    ["!Qwen/Qwen3.5-4B (rev. 851bf6e8)", "huggingface.co/Qwen/Qwen3.5-4B", "4.21B (4.66B with vision)",
     "main answering model (final system)", "Yes", "Yes, LoRA"],
    ["!Our LoRA adapter (rank 16)", "SinhalaMMLU-SharedTask/team-sinbotics, adapter/", "30.5M",
     "merged into Qwen3.5-4B", "Yes (merged)", "Yes, trained by us"],
    ["Claude Sonnet 5.5 (medium effort)", "Anthropic, via Claude Code", "n/a", "transcribing exam-paper page images for sinmulu_si; "
     "labelling answers without a marking scheme", "No", "No"],
    ["google/gemma-3-4b-it", "huggingface.co/google/gemma-3-4b-it", "4.30B",
     "phase-1 experiments and earlier submissions", "No", "Yes, LoRA (experiments)"],
    ["gemma-3-4b-pt, gemma-3-1b-it; NLLB-200-distilled-600M; MADLAD-400-3B", "huggingface.co (google, facebook)",
     "4.30B, 1.0B, 0.62B, 2.94B", "phase-1 screening only: references and Sinhala-English translation", "No", "No"],
])
fill(P[19], "Total inference parameters of the final system: 4.24B.",
     "That is the Qwen3.5-4B text model (4,205,751,296), which is what we load, plus the LoRA adapter (30,474,240). "
     "The adapter is merged into the weights, so the loaded model has 4,205,751,296 parameters. The full Qwen3.5-4B "
     "checkpoint, including the vision encoder we do not load, has 4,659,865,088. Nothing else is loaded at "
     "inference. Either count is at most 8B.")

# ------------------------------------------------------------------ 4. data
fill_table(T[2], [
    ["Shared-task public training set", "organisers", "Sinhala", "1,851 (Easy)", "validation only (phase-1 method choice); 70 as few-shot pool in phase-1 experiments",
     "18 near-duplicates of test questions removed"],
    ["Shared-task test set", "organisers", "Sinhala", "2,238", "predictions; 300-question subset scored with an answer key to choose between systems (declaration)", "none"],
    ["Global-MMLU, Sinhala", "huggingface.co/datasets/CohereLabs/Global-MMLU (si, rev. 0e619dbe)",
     "Sinhala (translated)", "14,324", "LoRA training (12,926) and validation (1,398)",
     "test and dev splits merged; 3 items with an empty option dropped"],
    ["sinmulu_si (ours)", "huggingface.co/datasets/JaniRaviRash-523/sinmulu-si; 3,486 exam-paper PDFs, transcribed with Claude Sonnet 5.5", "Sinhala", "15,180", "LoRA training (13,749) and validation (1,431)",
     "165 test overlaps and 10 duplicates removed"],
    ["!Combined (final adapter)", "union of the two above", "Sinhala", "29,504", "training 26,675 / validation "
     "2,829 (10% per subject, seed 42)", "questions over 1,024 tokens skipped (25,799 / 2,745 used)"],
])
fill(P[23], "How the data was used:",
     "Global-MMLU Sinhala and sinmulu_si are used for training and validation of the LoRA adapter. The shared-task "
     "public training set is used for validation (phase-1 method selection) and never for training. The test set "
     "is used only for predictions and for the system comparison described below.")
fill(P[24], "Synthetic or translated data:",
     "Global-MMLU Sinhala was translated from English by its authors; we use it as released. For sinmulu_si, most "
     "PDFs use legacy Sinhala fonts, so text extraction was unusable. Each page was rendered as an image and "
     "transcribed by Claude Sonnet 5.5, in batches by several agents working in parallel. Stems and options were "
     "copied verbatim. Only Sinhala-medium papers and true multiple-choice questions were kept. The prompt (Appendix "
     "D) also asked to skip calculation subjects, but this filter was not applied strictly: there are no Mathematics, "
     "Physics, Accounting or Statistics questions, yet 433 Chemistry, Engineering Technology and Science questions "
     "remain. Diagram-dependent or poorly scanned questions were dropped rather than guessed. Answers come from "
     "official marking schemes where available, matched by grade, year and province. Otherwise an answer was kept "
     "only if the model was highly confident, so some answers are model-derived. Records were validated, "
     "de-duplicated and cleaned.")
fill(P[25], "Overlap with the test set:",
     "We checked sinmulu_si (15,355 questions) against all 2,238 test questions. We found candidates with character "
     "n-gram TF-IDF, scored each pair with difflib similarity on the question and on the choices (the average of the "
     "3 best-matching choice pairs, order ignored), and reviewed the ranked pairs by hand. Round 1 (505 pairs ≥ 0.6) "
     "removed 143 questions; round 2 (≥ 0.7 on question, choices or both) removed 22 more, leaving 15,190. 260 test "
     "questions still have a question match above 0.7, and we judged most of them to be different template "
     "questions. 15 test questions still score ≥ 0.99 on question similarity. So the cleaning relies on manual "
     "judgement and does not prove zero leakage. Global-MMLU is translated MMLU and not derived from Sinhala exams.")
after(P[28], "Not ticked: see the exceptions below.", label=None)
fill(P[29], "Exceptions:",
     "(1) **Validation.** The shared-task public training set was used to choose prompts and methods in phase 1. "
     "The final prompt was selected this way, but contains no dataset question; 70 of its questions served as "
     "few-shot examples in phase-1 experiments only. (2) **Test subset.** To choose between the phase-2 systems "
     "(base model, adapter, thinking mode), full test-prediction files were scored on a stratified 300-question "
     "subset of the test set with an answer key for those questions (Section 7). No test question or answer was "
     "used for training. (3) **Possible indirect overlap.** sinmulu_si comes from Sri Lankan exam papers, as "
     "Sinhala MMLU does. Against the public training set, 1 sinmulu_si question has ≥ 0.5 character-5-gram "
     "similarity and 10 have ≥ 0.35; these were not removed. For the test set, see the overlap check above.")
fill(P[31], "Released on Hugging Face:",
     "Yes: code, inference script, predictions and the adapter at huggingface.co/SinhalaMMLU-SharedTask/team-sinbotics. "
     "Global-MMLU is public (linked above). sinmulu_si: huggingface.co/datasets/JaniRaviRash-523/sinmulu-si.")

# ------------------------------------------------------------------ 5. implementation
fill_table(T[3], [
    ["Method", "LoRA on the bf16 base model (no quantisation), merged with merge_and_unload after training"],
    ["Key hyperparameters", "rank 16, alpha 32, dropout 0.05; AdamW, lr 1e-4, cosine schedule with 3% warm-up, "
     "weight decay 0, gradient clipping 1.0; 1 epoch = 807 steps; 32 questions per step in micro-batches of at most "
     "12,288 padded tokens; max length 1,024 tokens; seed 42. Targets: q/k/v/o, gate/up/down and the "
     "linear-attention in_proj_qkv/in_proj_z/out_proj of all 32 text layers"],
    ["Training objective and format", "Answer-only: the loss is log P(gold digit) at a single position, after a "
     "plain multiple-choice chat prompt whose reply is pre-filled with \"Answer: \" (Appendix A). There is no loss "
     "on the question or on an end-of-turn token, so the model is not taught to stop after a digit and the "
     "reasoning prompt keeps working"],
    ["Libraries and versions", "torch 2.8.0, transformers 5.18.0, peft 0.21.1, accelerate 1.15.0, "
     "flash-linear-attention 0.4.0, causal-conv1d 1.7.0; own training loop (sinmmlu/train_lora.py)"],
])
fill(P[36], "Prompt template:", "no system prompt. The user turn is:")
q = P[36]
for line in ["You are an expert Sri Lankan teacher of {subject}.",
             "The following is a multiple choice question in Sinhala about {subject}.",
             "Think step by step in English, then choose the correct or most suitable answer from the options 1, 2, "
             "3, 4.",
             "End your reply with \"Answer: <number>\".",
             "",
             "Question: {Sinhala question}",
             "1. {option 1}   2. {option 2}   3. {option 3}   4. {option 4}   (one option per line)"]:
    q = after(q, line, size=Pt(8.5), font="Consolas")
q = after(q, "It is rendered with Qwen's chat template (enable_thinking=True, fixed template date). {subject} is "
          "the English subject name from the test file.")
fill(P[37], "Language of instructions:", "English instructions and English reasoning; the question and options "
     "stay in Sinhala.")
fill(P[38], "Few-shot examples:", "none (zero-shot).")
fill(P[40], "Decoding:", "greedy (do_sample=False), at most 4,096 new tokens, in length-sorted batches of 48 with "
     "left padding; seed 42.")
fill(P[41], "Answer extraction:",
     "log-probabilities of the tokens 1–4. A stated \"Answer: N\" is cut from the reasoning, \"</think>\" is added if "
     "missing, and \"\\n\\nAnswer: \" is appended. The argmax over the four digit tokens is the answer, so it never "
     "depends on parsing free text. In the final run, 56.0% of the reasoning texts stated an answer, and 97.4% of "
     "those agree with the read-off digit.")
fill(P[42], "Other components:", "none: no ensemble, voting or self-consistency. If the reasoning hits the token "
     "limit, the same read-off is applied to the truncated reasoning.")

# ------------------------------------------------------------------ 6. compute
fill_table(T[4], [
    ["sinmulu_si extraction and answer labels", "Claude Sonnet 5.5, medium effort", "Claude Code subscription",
     "3,486 PDFs", "20 (subscription)"],
    ["Main training and inference runs", "1× RTX 3090 24 GB", "RunPod", "about 20 GPU-h", "–"],
    ["Final inference on the test set (thinking mode)", "1× A100 SXM4 40 GB", "RunPod", "about 5 GPU-h", "–"],
    ["!Total", "", "", "!about 25 GPU-h + subscription", "!20 (subscription)"],
])
P[43].paragraph_format.keep_with_next = True
fill(P[46], "Minimum hardware to run your final system:",
     "one GPU with bf16 support and 24 GB of memory (generation batch 48 needs about 24 GB; smaller GPUs work more "
     "slowly, because a batch that runs out of memory is split). About 5 hours for the 2,238 test questions on one "
     "A100 40 GB.")

# ------------------------------------------------------------------ 7. experiments
dash = "–"
fill_table(T[5], [
    ["1", "Gemma-3-4B-it, paper prompt, digit scoring (baseline)", "dev confirmation (1,483, Easy)", "38.2", dash,
     dash, dash],
    ["2", "+ English chain-of-thought, teacher role (cot_en_role), 512 tokens", "dev confirmation", "45.6", dash,
     dash, dash],
    ["3", "+ few-shot, calibration, option-order averaging, cloze scoring (each)", "dev screening (280)",
     "31.4–42.5", dash, dash, dash],
    ["4", "row 2 + translation (NLLB / MADLAD / self) or self-consistency", "dev screening (ref. 48.9)",
     "41.8–50.4", dash, dash, dash],
    ["5", "Gemma, cot_en_role, 1,024 tokens (submission 2)", "test subset (300)", "49", "46", "47", "47.3"],
    ["6", "+ LoRA on Global-MMLU si", "test subset", "53", "51", "47", "50.3"],
    ["7", "+ LoRA on sinmulu_si (2 epochs)", "test subset", "49", "53", "46", "49.3"],
    ["8", "Qwen3.5-4B, cot_en_role (no thinking)", "test subset", "59", "48", "57", "54.7"],
    ["9", "row 8 + subject hints and a NOT rule", "test subset", "46", "46", "36", "42.7"],
    ["10", "row 8 + LoRA on the combined data", "test subset", "56", "53", "56", "55.0"],
    ["11", "!row 10 + thinking mode, 4,096 tokens (final system)", "test subset", "!59", "!56", "!64", "!59.7"],
    ["12", "!final system", "test leaderboard", "!54.67", "!52.30", "!65.77", "!57.58"],
])
fill(P[50], "Effect of each component:",
     "English reasoning was the largest gain on the dev data (+7.4 points, Holm-corrected p = 1.2e-6); translation, "
     "calibration, few-shot, cloze scoring and self-consistency did not help. On the test subset, moving from Gemma "
     "to Qwen3.5-4B gave +7.4 (McNemar p = 0.04). The adapter raised our own validation accuracy from 44.4% to 54.8% "
     "(Appendix B), but made no measurable difference on the subset (55.0 vs 54.7, p = 1.0). Thinking mode added "
     "+4.7 (p = 0.17; mostly on Hard questions, +8). The final system beat submission 2 by +12.4 (p = 0.001). The "
     "subject-aware prompt variant lost 12 points: it lowered the share of finished answers.")
fill(P[51], "Validation setup:",
     "Phase 1 chose methods on the public training set: screening on 280 questions, then confirmation of the best "
     "candidates on 1,483 held-out questions, under a rule fixed in advance (a gain of at least 1 point with "
     "Holm-corrected exact McNemar p < 0.05). Adapters were validated on 10% of their own training data. For "
     "phase 2 we had no labelled Medium or Hard development data, so we chose between full test-prediction files "
     "on a level-balanced 300-question test subset scored with an answer key. Subset numbers are therefore "
     "optimistic and not independent estimates (95% interval about ±5.7 points overall, ±10 per level). The "
     "leaderboard Macro Avg of the final system (57.58) is 2.1 points below its subset score (59.7).")

# ------------------------------------------------------------------ 8. analysis
fill(P[53], "Error analysis:",
     "On the leaderboard, Hard is our strongest level (65.77) and Medium the weakest (52.30). On the test subset, the final system is strongest on social science (76%) and STEM (73%), and weakest on "
     "humanities (50%; religion, art, dance, music) and Sinhala language (44%). The biggest factor is finishing the "
     "reasoning. Questions whose reasoning stated an answer are 73.4% correct, against 42.0% when it hit the "
     "4,096-token limit (44% of questions). The gap is largest on Easy questions (75.8% vs 31.6%), where the model "
     "often over-thinks culture-specific recall questions. On dev data, sub-option, fill-in-the-blank and numeric "
     "questions are the weakest types (32–37%, against 46% on plain questions). Negated (\"which is NOT ...\") "
     "questions are handled well with Sinhala input, but badly after machine translation, which dropped the "
     "negation in about 40% of them.")
fill(P[54], "Limitations:",
     "System choice used test questions (see the declaration). sinmulu_si contains transcription and model-derived "
     "labelling errors that we could not measure, and its residual near-duplicates mean zero leakage is not proven. "
     "Long reasoning is slow (about 5 h on an A100) and often truncated. Sinhala costs about 4.8 Qwen tokens per "
     "word. bf16 results vary across GPUs: the same Gemma system scored 44.3% and 47.3% on two GPU types.")
fill(P[55], "What you would try next:",
     "Build a labelled Medium/Hard development set from our own exam papers, so that system choices no longer touch "
     "the test set. Then train the adapter on short English rationales rather than the digit alone, which should "
     "reduce truncated reasoning.")

# ------------------------------------------------------------------ 9. checklist
fill(P[58], None, "☒  All code, the inference script, requirements.txt and predictions.jsonl are in "
     "SinhalaMMLU-SharedTask/team-sinbotics")
fill(P[59], None, "☐  All trained models and adapters are in SinhalaMMLU-SharedTask/team-<teamname>-<model> "
     "repositories. (Our single adapter is in the adapter/ folder of SinhalaMMLU-SharedTask/team-sinbotics, with "
     "weights, config and tokenizer files; inference.py loads it from there.)")
fill(P[60], None, "☐  The inference script reads the dataset from the DATA_REPO line and runs from start to finish "
     "in a fresh environment. (It reads DATA_REPO and every .json file. A full fresh-environment run is "
     "{{confirm after the test run}}.)")
fill(P[61], None, "☒  All data used is released or linked in Section 4")
fill(P[62], None, "☐  Every model, API and cost is listed in Sections 3 and 6. (Models and the Claude Code "
     "subscription are listed; GPU rental costs are not.)")
fill(P[63], None, "☒  The report file is named SinhalaMMLU2026_team-sinbotics_SystemReport.pdf")

# ------------------------------------------------------------------ 10. AI assistants
fill(P[65], None,
     "Claude Code (Anthropic; model Claude Opus 5.5) wrote and ran most of the code, ran the experiments on rented "
     "GPUs, prepared the Hugging Face submission, and drafted this report and its tables from our logs. The team "
     "directed the work and made the decisions. Claude Sonnet 5.5 transcribed the exam-paper images of sinmulu_si "
     "and labelled answers where no marking scheme was available (Section 4). No AI assistant or API is used at "
     "inference.")

# ------------------------------------------------------------------ references
refs = [
    "Ashmari Pramodya, Nirasha Nelki, Heshan Shalinda, Chamila Liyanage, Yusuke Sakai, Randil Pushpananda, Ruvan "
    "Weerasinghe, Hidetaka Kamigaito, and Taro Watanabe. 2025. SinhalaMMLU: A Comprehensive Benchmark for Evaluating "
    "Multitask Language Understanding in Sinhala. In Proceedings of EMNLP 2025.",
    "Shivalika Singh et al. 2025. Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in "
    "Multilingual Evaluation. In Proceedings of ACL 2025.",
    "Qwen Team. 2026. Qwen3.5-4B. Model card, https://huggingface.co/Qwen/Qwen3.5-4B.",
    "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "
    "2022. LoRA: Low-Rank Adaptation of Large Language Models. In Proceedings of ICLR 2022.",
    "Gemma Team. 2025. Gemma 3 Technical Report. arXiv:2503.19786.",
    "NLLB Team. 2022. No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672.",
    "Sneha Kudugunta et al. 2023. MADLAD-400: A Multilingual and Document-Level Large Audited Dataset. In NeurIPS "
    "2023 Datasets and Benchmarks.",
    "Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate Before Use: Improving Few-Shot "
    "Performance of Language Models. In Proceedings of ICML 2021.",
    "Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. "
    "2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings of ICLR 2023.",
    "Thomas Wolf et al. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of EMNLP "
    "2020: System Demonstrations.",
    "Sourab Mangrulkar et al. 2022. PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods. "
    "https://github.com/huggingface/peft.",
    "Songlin Yang and Yu Zhang. 2024. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear "
    "Attention Mechanism. https://github.com/fla-org/flash-linear-attention.",
    "Anthropic. 2026. Claude Sonnet 5.5 (used via Claude Code for data transcription and answer labelling only).",
]
fill(P[67], None, refs[0])
chain(P[67], refs[1:])

# ------------------------------------------------------------------ appendix
fill(P[69], "A. Training prompt (en_instr_nosubject)", "for the LoRA adapter; the chat reply is pre-filled with "
     "\"Answer: \" and the loss is on the next token only:")
q = P[69]
for line in ["The following is a multiple choice question in Sinhala.",
             "Choose the correct or most suitable answer from the options 1, 2, 3, 4.",
             "", "Question: {Sinhala question}", "1. {option 1} ... 4. {option 4}", "Answer:"]:
    q = after(q, line, size=Pt(8.5), font="Consolas")
q = after(q, "B. Validation accuracy of the final adapter during training (2,745 held-out questions, digit prompt)",
          label=None, style_from=P[69])
q.runs[0].bold = True
q = table_after(q, T[3], ["Step", "Accuracy (%)"], [
    ["0 / 100 / 200 / 300", "44.4 / 47.4 / 49.9 / 51.7"],
    ["400 / 500 / 600 / 700 / 807", "51.7 / 52.8 / 53.2 / 54.5 / 54.8"],
])
q = after(q, "Gemma-3-4B-it adapters, the same validation method: Global-MMLU 40.1 → 41.5; sinmulu_si (rank 16) "
          "40.9 → 47.1 / 47.3 / 45.8 after epochs 1 / 2 / 3; sinmulu_si (rank 8, dropout 0.1) 37.5 → 43.8 / 44.4 "
          "after epochs 1 / 2. Every run peaked after one epoch.")
q = after(q, "C. Phase-1 results (Gemma-3-4B-it, accuracy %, screening 280 / confirmation 1,483 public training "
          "questions)")
q.runs[0].bold = True
q = table_after(q, T[3], ["Condition", "Screening / confirmation"], [
    ["Paper prompt, Sinhala, no subject (baseline)", "37.5 / 38.2"],
    ["Paper prompt + subject; teacher role + subject (Sinhala)", "42.1 / –; 43.6 / –"],
    ["English instruction; 3-shot same subject", "41.8 / –; 42.5 / –"],
    ["Contextual calibration; cloze scoring of option texts", "34.3 / –; 31.4 / –"],
    ["Reasoning in Sinhala; reasoning in English", "43.2 / –; 47.9 / 44.9"],
    ["English reasoning, teacher role, 512 / 1,024 tokens", "48.2 / 45.6; 48.9 / 45.6"],
    ["+ 2 worked examples; ensemble of 3 reasoning prompts", "50.7 / 44.8; 50.4 / 46.7"],
    ["+ negation check (pre-registered +2 rule met on screening only)", "51.4 / 44.3"],
    ["+ option-order averaging (2 orders); self-consistency (5 samples)", "49.6 / –; 47.5 / –"],
    ["+ bilingual input (MADLAD / NLLB); English-only input (MADLAD)", "49.3 / –, 47.9 / –; 44.3 / –"],
    ["Gemma-3-1B-it, paper prompt", "25.7 / –"],
])
q = after(q, "D. Transcription prompt for sinmulu_si (Claude Sonnet 5.5)")
q.runs[0].bold = True
q = after(q, "The prompt is given verbatim. Three things differ from it in the released data. Marking-scheme matching, "
          "dropping diagram-dependent questions and the final cleaning were not part of this prompt. The released "
          "file keeps four-option questions only, and omits the source_file and තර්කනය (rationale) fields, so "
          "rationales were not used in training. The subject filter was not applied strictly (Section 4).")
with open("transcription_prompt.txt", encoding="utf-8") as f:
    for line in f.read().splitlines():
        q = after(q, line, size=Pt(8), font="Consolas")

d.save(out)
print("saved", out)
