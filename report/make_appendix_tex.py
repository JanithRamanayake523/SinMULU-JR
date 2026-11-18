"""Copy the long tables of ../report.tex into appendix_tables.tex, so the appendix and the long report stay identical.
python make_appendix_tex.py"""

with open("../report.tex", encoding="utf-8") as f:
    lines = f.read().split("\n")


def block(first, last):
    """Lines between the first line containing `first` and the next line containing `last`, inclusive."""
    i = next(n for n, s in enumerate(lines) if first in s)
    j = next(n for n, s in enumerate(lines) if n >= i and last in s)
    return "\n".join(lines[i:j + 1])


parts = [
    "\\section{Prompts}\\label{app:prompts}",
    block("Shared pieces (Appendix~B", "\\end{longtable}}"),
    "\\section{Screening Results}\\label{app:screen}",
    block("{\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\end{longtable}}"),
    "\\section{Accuracy per Subject}\\label{app:subjects}",
    block("Subject & & $n$ & A0", "\\end{tabular}"),
]
text = "\n\n".join(parts).replace("\\cite{pramodya2025}", "\\citep{pramodya2025sinhalammlu}")
text = text.replace("(phase 0)", "(phase 1)").replace("Screening on \\texttt{inference\\_subset}", "Screening set 1")

# prompts added after the long report's table (wording from sinmmlu/prompts.py)
EXTRA_PROMPTS = r"""\cond{cot\_en\_role\_think} (\textbf{final system}) & the same text as \cond{cot\_en\_role}, rendered with \texttt{enable\_thinking=True}, at most 4,096 new tokens (Section~\ref{sec:prompting})\\
\cond{cot\_en\_role\_subject\_aware} & \textsc{role-head} / one subject hint (listed below) / ``If the question asks which option is NOT correct (or does not match), choose the false option.'' / \textsc{cot} / / Question: \{Sinhala question\} / options\\
\cond{cot\_en\_role\_bilingual} & \textsc{role-head} / ``The question is given in Sinhala, followed by a machine translation into English. The translation may contain errors; the Sinhala text is the original.'' / \textsc{cot} / / Question (Sinhala): \{Sinhala question\} / options / / Machine translation (English): / Question: \{English translation\} / translated options\\
\cond{cot\_en\_role\_en} & ``You are an expert Sri Lankan teacher of \{subject\}. / The following is a multiple choice question about \{subject\}.'' / \textsc{cot} / / Question: \{English translation\} / translated options\\
\cond{cot\_en\_role\_negcheck} & \textsc{role-head} / ``Before reasoning, state whether the question asks for the correct option or for the option that is NOT correct, does NOT match, or is false. Then think step by step in English and choose the correct or most suitable answer from the options 1, 2, 3, 4. / End your reply with ``Answer: <number>''.'' / / Question: \{Sinhala question\} / options\\
\cond{cot\_en\_role\_each\_option} & \textsc{role-head} / ``Evaluate each of the four options one by one and say whether it is correct or incorrect for this question, thinking in English. Then choose the correct or most suitable answer from the options 1, 2, 3, 4. / End your reply with ``Answer: <number>''.'' / / Question: \{Sinhala question\} / options\\
\cond{cot\_en\_role\_translate\_first} & \textsc{role-head} / ``First translate the question and the four options into English. Then think step by step in English and choose the correct or most suitable answer from the options 1, 2, 3, 4. / End your reply with ``Answer: <number>''.'' / / Question: \{Sinhala question\} / options\\
\cond{cot\_en\_role\_negchain}, turn 1 & \textsc{role-head} / ``Think step by step in English. For each of the options 1, 2, 3, 4, decide whether it is a correct answer to this question. More than one option may be correct. / End your reply with ``Correct options: <numbers separated by commas>''.'' / / Question: \{Sinhala question\} / options\\
\cond{cot\_en\_role\_negchain}, turn 2 & ``Now negate the question: write the opposite question in English (if the question asks which option is correct, the opposite question asks which options are NOT correct, and if it asks which option is NOT correct, the opposite question asks which options are correct). Then think step by step in English and decide for each of the options 1, 2, 3, 4 whether it answers the opposite question. / End your reply with ``Options for the opposite question: <numbers separated by commas>''.''\\
\cond{cot\_en\_role\_negchain}, turn 3 & ``Now use both analyses, the original question with its correct options and the opposite question with its options, to choose the single correct or most suitable answer to the ORIGINAL question from the options 1, 2, 3, 4. Explain briefly in English (at most three sentences), then end your reply with ``Answer: <number>''.''\\
"""
PROMPT_KEYS = r"""
\noindent The later prompts use two more shared pieces: \textsc{role-head} = ``You are an expert Sri Lankan teacher of \{subject\}. / The following is a multiple choice question in Sinhala about \{subject\}.''; \textsc{cot} = ``Think step by step in English, then choose the correct or most suitable answer from the options 1, 2, 3, 4. / End your reply with ``Answer: <number>''.''

\noindent Subject hints of \cond{cot\_en\_role\_subject\_aware}, by subject group. Sinhala: ``This question tests Sinhala language and literature (grammar, vocabulary, idioms, literary works). Pay close attention to the exact Sinhala wording and spelling of each option.'' Religion: ``Answer according to the teachings of this religion as taught in the Sri Lankan school curriculum.'' History, geography, civics, politics, communication, health: ``Use knowledge of Sri Lanka (history, geography, government and society) as taught in the Sri Lankan school curriculum.'' Science and technology: ``Reason step by step and check facts, units and calculations carefully.'' Business and economics: ``Apply the definitions and principles of the Sri Lankan syllabus and check any calculation.'' Art, dance, drama, music: ``Recall the specific traditional Sri Lankan terms, artists, forms and techniques.''
"""
text = text.replace("\\cond{cot\\_en\\_role} (final)", "\\cond{cot\\_en\\_role} (phase-2 winner)", 1)
text = text.replace("{p{0.24\\linewidth}p{0.70\\linewidth}}", "{p{0.29\\linewidth}p{0.65\\linewidth}}", 1)
marker = "A0 with the options replaced by their English translations\\\\\n"
assert marker in text
text = text.replace(marker, marker + EXTRA_PROMPTS, 1)
end = text.index("\\end{longtable}}", text.index(marker)) + len("\\end{longtable}}")
text = text[:end] + "\n" + PROMPT_KEYS + text[end:]

# the per-subject table: restore the opening lost by starting at its header row
text = text.replace("Subject & & $n$ & A0",
                    "\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
                    "\\begin{tabular}{llrrrrrr}\n\\toprule\nSubject & & $n$ & A0", 1)
text += ("\n\\end{center}\n\n"
         "\\noindent Descriptive only: per-subject intervals are $\\pm$8--12 points, so no decision was "
         "taken per subject (\\texttt{cot\\_en\\_fs2} is \\texttt{cot\\_en\\_fewshot2}).\n\n"
         "\\clearpage")

with open("appendix_tables.tex", "w", encoding="utf-8") as g:
    g.write(text + "\n")
