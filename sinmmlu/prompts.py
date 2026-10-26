"""Prompt registry. A prompt name fully determines what the model sees and how it is scored.

build(row, name) returns the conversation as a tuple of (role, text) turns; the last turn is the user turn being
answered. Zero-shot prompts are one user turn; k-shot prompts put k solved examples first (user turn + assistant
answer). model.render() turns this into the chat template, or into plain text for raw (non-chat) scoring.

Each Prompt also says
  method   digits : next-token probability of the option labels after the prompt
           cloze  : length-normalised log-likelihood of each option's text (options are not shown)
           cot    : the model writes its reasoning first; the labels are then scored after `cue`
  labels   option labels shown and scored ("1".."4" or "A".."D")
  lang     which text the prompt shows: si (original), en (translated), both, opts_en (translated options only);
           anything but si needs a translator (python -m sinmmlu.translate)
"""
from dataclasses import dataclass
from typing import Callable, Optional

from .data import load_jsonl, norm_model, path, subject_en, subject_family, subject_si

DIGITS = ("1", "2", "3", "4")
LETTERS = ("A", "B", "C", "D")
FEWSHOT_POOL = "dev/fewshot_pool.jsonl"
RATIONALE_POOL = "dev/fewshot_cot_en.jsonl"  # pool questions with a correct, model-written English rationale

# Appendix B of Pramodya et al. (2025)
SI_HEAD_SUBJECT = "පහත දැක්වෙන්නේ {subject} සම්බන්ධ බහුවරණ ප්‍රශ්නයකි.\n"
SI_HEAD = "පහත දැක්වෙන්නේ බහුවරණ ප්‍රශ්නයකි.\n"
SI_INSTR = "පහත දැක්වෙන ප්‍රශ්නය සඳහා නිවැරදි හෝ වඩාත්ම ගැළපෙන පිළිතුර 1, 2, 3, 4 පිළිතුරුවලින් තෝරන්න.\n"
SI_INSTR_LETTERS = "පහත දැක්වෙන ප්‍රශ්නය සඳහා නිවැරදි හෝ වඩාත්ම ගැළපෙන පිළිතුර A, B, C, D පිළිතුරුවලින් තෝරන්න.\n"
# "As your answer, write only the digit 1, 2, 3 or 4. Do not write anything else."
SI_STRICT = "ඔබේ පිළිතුර ලෙස 1, 2, 3 හෝ 4 යන ඉලක්කම පමණක් ලියන්න. වෙනත් කිසිවක් ලියන්න එපා.\n"
# "You are an expert Sri Lankan teacher of the subject {subject}."
SI_ROLE = "ඔබ {subject} විෂය පිළිබඳ ප්‍රවීණ ශ්‍රී ලාංකික ගුරුවරයෙකි.\n"
# "Before choosing the answer, reason step by step in Sinhala. At the end write "Answer: <number>"."
SI_COT = "පිළිතුර තෝරා ගැනීමට පෙර පියවරෙන් පියවර සිංහලෙන් තර්ක කරන්න. අවසානයේ \"පිළිතුර: <අංකය>\" ලෙස ලියන්න.\n"
# "The following is a question. Write the correct answer."
SI_CLOZE_HEAD = "පහත දැක්වෙන්නේ ප්‍රශ්නයකි. නිවැරදි පිළිතුර ලියන්න.\n"
SI_Q, SI_A = "ප්‍රශ්නය", "පිළිතුර"

EN_HEAD = "The following is a multiple choice question.\n"
EN_HEAD_SI = "The following is a multiple choice question in Sinhala.\n"
EN_INSTR = "Choose the correct or most suitable answer from the options 1, 2, 3, 4.\n"
EN_COT = "Think step by step in English, then choose the correct or most suitable answer from the options 1, 2, 3, 4.\n" \
         "End your reply with \"Answer: <number>\".\n"
EN_HEAD_SI_SUBJECT = "The following is a multiple choice question in Sinhala about {subject}.\n"
EN_ROLE = "You are an expert Sri Lankan teacher of {subject}.\n"
EN_HEAD_SUBJECT = "The following is a multiple choice question about {subject}.\n"
EN_BILINGUAL_COT = ("The question is given in Sinhala, followed by a machine translation into English. "
                    "The translation may contain errors; the Sinhala text is the original.\n")
EN_NEG_CHECK = ("Before reasoning, state whether the question asks for the correct option or for the option that is NOT "
                "correct, does NOT match, or is false. Then think step by step in English and choose the correct or most "
                "suitable answer from the options 1, 2, 3, 4.\nEnd your reply with \"Answer: <number>\".\n")
EN_EACH_OPTION = ("Evaluate each of the four options one by one and say whether it is correct or incorrect for this "
                  "question, thinking in English. Then choose the correct or most suitable answer from the options "
                  "1, 2, 3, 4.\nEnd your reply with \"Answer: <number>\".\n")
# negation-consistency chain (2026-10-26): all correct options -> opposite question -> one final answer
EN_ALL_CORRECT = ("Think step by step in English. For each of the options 1, 2, 3, 4, decide whether it is a correct "
                  "answer to this question. More than one option may be correct.\n"
                  "End your reply with \"Correct options: <numbers separated by commas>\".\n")
EN_NEGATED_STEP = ("Now negate the question: write the opposite question in English (if the question asks which option "
                   "is correct, the opposite question asks which options are NOT correct, and if it asks which option is "
                   "NOT correct, the opposite question asks which options are correct). Then think step by step in "
                   "English and decide for each of the options 1, 2, 3, 4 whether it answers the opposite question.\n"
                   "End your reply with \"Options for the opposite question: <numbers separated by commas>\".")
EN_FINAL_STEP = ("Now use both analyses, the original question with its correct options and the opposite question with "
                 "its options, to choose the single correct or most suitable answer to the ORIGINAL question from the "
                 "options 1, 2, 3, 4. Explain briefly in English (at most three sentences), then end your reply with "
                 "\"Answer: <number>\".")
EN_TRANSLATE_FIRST = ("First translate the question and the four options into English. Then think step by step in "
                      "English and choose the correct or most suitable answer from the options 1, 2, 3, 4.\n"
                      "End your reply with \"Answer: <number>\".\n")
EN_BILINGUAL = ("The following is a multiple choice question in Sinhala, followed by a machine translation into English.\n"
                "The translation may contain errors; the Sinhala text is the original.\n")


def options(choices, labels=DIGITS):
    return "\n".join(f"{l}. {norm_model(c)}" for l, c in zip(labels, choices))


def q_si(r):
    return norm_model(r["question"])


def q_en(r):
    return norm_model(r["question_en"])


def body(question, opts, q_word=SI_Q, a_word=SI_A):
    return f"\n{q_word}: {question}\n{opts}\n{a_word}:"


@dataclass(frozen=True)
class Prompt:
    text: Callable[[dict], str]                      # row -> user text of the question being answered
    subject: Optional[Callable[[dict], str]] = None  # subject shown in the prompt (calibration prior key)
    method: str = "digits"
    labels: tuple = DIGITS
    lang: str = "si"
    shots: int = 0
    answer_word: str = SI_A                          # pre-filled answer marker: the reply starts "<answer_word>: "
    cue: str = ""                                    # cot: text appended after the reasoning, before scoring
    max_new_tokens: int = 0                          # cot: reasoning length limit
    followups: tuple = ()                            # chain: further user turns, each (text, max_new_tokens)
    pool: str = FEWSHOT_POOL                         # where k-shot examples come from
    example_reply: Optional[Callable[[dict], str]] = None  # assistant turn of an example (default: its answer label)


def _paper(r):
    return SI_HEAD + SI_INSTR + body(q_si(r), options(r["choices"]))


def _bilingual(r):
    return (EN_BILINGUAL + EN_INSTR + f"\nSinhala:\n{SI_Q}: {q_si(r)}\n{options(r['choices'])}\n"
            f"\nEnglish:\nQuestion: {q_en(r)}\n{options(r['choices_en'])}\nAnswer:")


def _paper_subject_si(r):
    return SI_HEAD_SUBJECT.format(subject=subject_si(r)) + SI_INSTR + body(q_si(r), options(r["choices"]))


def _cot_en(r):
    return EN_HEAD_SI + EN_COT + f"\nQuestion: {q_si(r)}\n{options(r['choices'])}"


def _cot_en_subject(r):
    return EN_HEAD_SI_SUBJECT.format(subject=subject_en(r)) + EN_COT + f"\nQuestion: {q_si(r)}\n{options(r['choices'])}"


COT_EN = dict(method="cot", cue="\n\nAnswer: ", answer_word="Answer", max_new_tokens=512)

PROMPTS = {
    # A0 and prompt wording (B1-B6)
    "paper_nosubject": Prompt(_paper),
    "paper_subject_si": Prompt(_paper_subject_si, subject=subject_si),
    "paper_subject_en": Prompt(lambda r: SI_HEAD_SUBJECT.format(subject=subject_en(r)) + SI_INSTR
                               + body(q_si(r), options(r["choices"])), subject=subject_en),
    "strict_nosubject": Prompt(lambda r: SI_HEAD + SI_INSTR + SI_STRICT + body(q_si(r), options(r["choices"]))),
    "role_subject_si": Prompt(lambda r: SI_ROLE.format(subject=subject_si(r)) + SI_HEAD_SUBJECT.format(subject=subject_si(r))
                              + SI_INSTR + body(q_si(r), options(r["choices"])), subject=subject_si),
    "en_instr_nosubject": Prompt(lambda r: EN_HEAD_SI + EN_INSTR + body(q_si(r), options(r["choices"]), "Question", "Answer"),
                                 answer_word="Answer"),
    "letters_nosubject": Prompt(lambda r: SI_HEAD + SI_INSTR_LETTERS + body(q_si(r), options(r["choices"], LETTERS)),
                                labels=LETTERS),
    # few-shot (B7): k fixed solved examples of the same subject from dev/fewshot_pool.jsonl
    **{f"fewshot{k}_nosubject": Prompt(_paper, shots=k) for k in (1, 3, 5)},
    "fewshot3_subject_si": Prompt(_paper_subject_si, subject=subject_si, shots=3),
    # scoring alternatives (C4) and reasoning (C6, C7)
    "cloze_nosubject": Prompt(lambda r: SI_CLOZE_HEAD + f"\n{SI_Q}: {q_si(r)}\n{SI_A}:", method="cloze"),
    "cot_si": Prompt(lambda r: SI_HEAD + SI_INSTR + SI_COT + f"\n{SI_Q}: {q_si(r)}\n{options(r['choices'])}",
                     method="cot", cue=f"\n\n{SI_A}: ", max_new_tokens=768),
    "cot_en": Prompt(_cot_en, **COT_EN),
    # combinations after wave 3: English reasoning + subject / teacher role / 2 worked examples with model rationales
    "cot_en_subject": Prompt(_cot_en_subject, subject=subject_en, **COT_EN),
    "cot_en_role": Prompt(lambda r: EN_ROLE.format(subject=subject_en(r)) + _cot_en_subject(r), subject=subject_en,
                          **COT_EN),
    "cot_en_fewshot2": Prompt(_cot_en, shots=2, pool=RATIONALE_POOL, example_reply=lambda e: e["rationale"], **COT_EN),
    # translation + reasoning (2026-10-24 evening): the winner's teacher-role reasoning with translated input
    "cot_en_role_bilingual": Prompt(
        lambda r: EN_ROLE.format(subject=subject_en(r)) + EN_HEAD_SI_SUBJECT.format(subject=subject_en(r))
        + EN_BILINGUAL_COT + EN_COT + f"\nQuestion (Sinhala): {q_si(r)}\n{options(r['choices'])}\n"
        f"\nMachine translation (English):\nQuestion: {q_en(r)}\n{options(r['choices_en'])}",
        subject=subject_en, lang="both", **COT_EN),
    "cot_en_role_en": Prompt(
        lambda r: EN_ROLE.format(subject=subject_en(r)) + EN_HEAD_SUBJECT.format(subject=subject_en(r)) + EN_COT
        + f"\nQuestion: {q_en(r)}\n{options(r['choices_en'])}",
        subject=subject_en, lang="en", **COT_EN),
    # question-type handling (2026-10-24 evening): notice negation first / judge every option
    "cot_en_role_negcheck": Prompt(
        lambda r: EN_ROLE.format(subject=subject_en(r)) + EN_HEAD_SI_SUBJECT.format(subject=subject_en(r))
        + EN_NEG_CHECK + f"\nQuestion: {q_si(r)}\n{options(r['choices'])}",
        subject=subject_en, **COT_EN),
    "cot_en_role_each_option": Prompt(
        lambda r: EN_ROLE.format(subject=subject_en(r)) + EN_HEAD_SI_SUBJECT.format(subject=subject_en(r))
        + EN_EACH_OPTION + f"\nQuestion: {q_si(r)}\n{options(r['choices'])}",
        subject=subject_en, **COT_EN),
    "cot_en_role_negchain": Prompt(
        lambda r: EN_ROLE.format(subject=subject_en(r)) + EN_HEAD_SI_SUBJECT.format(subject=subject_en(r))
        + EN_ALL_CORRECT + f"\nQuestion: {q_si(r)}\n{options(r['choices'])}",
        subject=subject_en, method="chain", followups=((EN_NEGATED_STEP, 1024), (EN_FINAL_STEP, 256)),
        cue="\n\nAnswer: ", answer_word="Answer", max_new_tokens=1024),
    "cot_en_role_translate_first": Prompt(
        lambda r: EN_ROLE.format(subject=subject_en(r)) + EN_HEAD_SI_SUBJECT.format(subject=subject_en(r))
        + EN_TRANSLATE_FIRST + f"\nQuestion: {q_si(r)}\n{options(r['choices'])}",
        subject=subject_en, **COT_EN),
    # translation (D1-D5): the translator is part of the condition
    "en_nosubject": Prompt(lambda r: EN_HEAD + EN_INSTR + body(q_en(r), options(r["choices_en"]), "Question", "Answer"),
                           lang="en", answer_word="Answer"),
    "bilingual_nosubject": Prompt(_bilingual, lang="both", answer_word="Answer"),
    "si_q_en_opts": Prompt(lambda r: SI_HEAD + SI_INSTR + body(q_si(r), options(r["choices_en"])), lang="opts_en"),
}


# ----------------------------------------------------------------------------- few-shot examples
_pools = {}


def _load_pool(src=FEWSHOT_POOL):
    if src not in _pools:
        if not path(src).exists():
            raise FileNotFoundError(f"{src} missing: run python -m sinmmlu.build_dev "
                                    "(rationales: python -m sinmmlu.rationales)")
        by = {}
        for r in load_jsonl(src):
            by.setdefault(subject_family(r), []).append(r)
        # generic fallback for subjects without (enough) examples, e.g. most Medium/Hard test subjects: the first
        # example of each subject, in a fixed order
        generic = [v[0] for _, v in sorted(by.items())]
        _pools[src] = (by, generic)
    return _pools[src]


def examples(row, k, src=FEWSHOT_POOL):
    by, generic = _load_pool(src)
    pool = [e for e in by.get(subject_family(row), []) if e["uid"] != row.get("uid")]
    if len(pool) < k:
        pool += [e for e in generic if e["uid"] != row.get("uid") and e not in pool]
    return pool[:k]


# ----------------------------------------------------------------------------- build
def build(row, name):
    p = PROMPTS[name]
    turns = []
    for e in examples(row, p.shots, p.pool) if p.shots else ():
        reply = p.example_reply(e) if p.example_reply else p.labels[int(e["answer"]) - 1]
        turns += [("user", p.text(e)), ("assistant", reply)]
    turns.append(("user", p.text(row)))
    return tuple(turns)


def content_free(rep, text):
    """Calibration input: `rep`'s subject and metadata, with the question and every option replaced by `text`."""
    filler = [text or "-"] * 4
    return dict(rep, question=text, choices=filler, question_en=text, choices_en=filler)
