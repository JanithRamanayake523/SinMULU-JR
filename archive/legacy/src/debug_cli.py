"""Command-line version of colab_debug_next_token.ipynb (for a cluster without notebooks). DEV DATA ONLY.

Prints the same sections as the notebook; redirect stdout to a file (the job scripts do this).
Examples
  python src/debug_cli.py --model-key qwen --prompt nosubject_strict --chat on --perms 1 --n 20
  python src/debug_cli.py --model-key gemma --prompt nosubject --chat off --perms 4 --calibrate true --n 20 --sections scoring,detail
  python src/debug_cli.py --compare-only          # tokenisation of one question through all three tokenizers (no model weights)
Sections (comma separated, default: tokenization,scoring,detail): tokenization, scoring, detail, summary
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import debug_next_token as D
from data_utils import ROOT, load_jsonl
from eval_logprob import parse_bool

ALL_TOKENIZERS = ["google/gemma-3-4b-it", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="gemma4b", help="gemma4b | llama3b | qwen4b | qwen | gemma | llama (1B models) | a config name in configs/")
    ap.add_argument("--prompt", default="nosubject_strict", choices=["nosubject", "nosubject_strict", "subject_si", "subject_en"])
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"], help="auto = the model's default way: on if chat capable, else off")
    ap.add_argument("--perms", type=int, default=1, choices=[1, 4])
    ap.add_argument("--calibrate", type=parse_bool, default=False)
    ap.add_argument("--n", type=int, default=20, help="first N dev questions for scoring / detail / summary")
    ap.add_argument("--question-index", type=int, default=0, help="question used in the tokenisation section")
    ap.add_argument("--perm-index", type=int, default=0, choices=[0, 1, 2, 3], help="option order to inspect (0 = original)")
    ap.add_argument("--generate", type=int, default=100, help="greedy continuation tokens per question (0 = off)")
    ap.add_argument("--sections", default="tokenization,scoring,detail")
    ap.add_argument("--no-compare", action="store_true", help="skip section 4d (the three-tokenizer comparison)")
    ap.add_argument("--compare-only", action="store_true", help="only compare the three tokenizers (no model weights)")
    a = ap.parse_args()

    rows = load_jsonl("dev/dev_easy_full.jsonl")
    row = rows[a.question_index]
    if a.compare_only:
        print("Question", a.question_index, row["subject"])
        D.compare_tokenizers(ALL_TOKENIZERS, row["question"], n_words=10)
        return
    P = D.preset(a.model_key)
    chat = P["chat_capable"] if a.chat == "auto" else a.chat == "on"
    a.chat = "on" if chat else "off"  # resolved value, used in file names and printouts
    if chat and not P["chat_capable"]:
        sys.exit(f"{P['model']} is not chat capable: use --chat off")
    sections = set(a.sections.split(","))
    tok, model = D.load(P["model"], P["dtype"])
    print(f"MODEL {P['model']} dtype={P['dtype']} | prompt={a.prompt} chat={a.chat} perms={a.perms} calibrate={a.calibrate} "
          f"| N={a.n} question_index={a.question_index} perm_index={a.perm_index}")
    print(f"Question {a.question_index} | {row['subject']} | gold answer: {row['answer']}")
    print(row["question"])
    for i, c in enumerate(row["choices"], 1):
        print(i, c)

    if "tokenization" in sections:
        print("\n" + "=" * 100 + "\n4a. THE EXACT PROMPT, TOKEN BY TOKEN")
        D.show_prompt_tokens(tok, row, prompt_mode=a.prompt, chat=chat, perm_index=a.perm_index)
        print("\n" + "=" * 100 + "\n4b. WORD BY WORD (fertility)")
        D.word_view(tok, row["question"], n_words=20)
        print("\n" + "=" * 100 + "\n4c. THE ANSWER TOKENS THE SCORER LOOKS FOR")
        D.answer_tokenization(tok)
        if not a.no_compare:
            print("\n" + "=" * 100 + "\n4d. THE SAME QUESTION THROUGH ALL THREE TOKENIZERS")
            D.compare_tokenizers(ALL_TOKENIZERS, row["question"], n_words=10)
    if "scoring" in sections:
        print("\n" + "=" * 100 + f"\n5. SCORING TABLE (first {a.n} questions, real scoring path)")
        df = D.scoring_table(tok, model, rows, n=a.n, prompt_mode=a.prompt, chat=chat, perms=a.perms, calibrate=a.calibrate)
        print(df.to_string(index=False))
        out = ROOT / "outputs" / "debug"
        out.mkdir(parents=True, exist_ok=True)
        name = f"{a.model_key}_{a.prompt}_chat{a.chat}_p{a.perms}_c{int(a.calibrate)}_n{a.n}.csv"
        df.to_csv(out / name, index=False, encoding="utf-8-sig")
        print("saved", out / name)
    # Token generation (greedy continuation) only for perms=1 and no calibration (chat on or off): it shows what the
    # model writes after the plain prompt. With permutation averaging or calibration it is skipped.
    gen = a.generate if (a.perms == 1 and not a.calibrate) else 0
    if a.generate and gen == 0:
        print("\n(greedy continuation skipped: it is only produced for perms=1 without calibration)")
    if "detail" in sections:
        print("\n" + "=" * 100 + f"\n6. NEXT-TOKEN DETAIL PER QUESTION (first {a.n}, option order {a.perm_index})")
        for i, r in enumerate(rows[:a.n]):
            print("#" * 100)
            print(f"QUESTION {i}  |  {r['subject']}  |  gold answer (original order): {r['answer']}")
            D.show_question(tok, model, r, prompt_mode=a.prompt, chat=chat, perm_index=a.perm_index, calibrate=a.calibrate, generate=gen)
    if "summary" in sections:
        print("\n" + "=" * 100 + f"\n7. SUMMARY over the first {a.n} questions")
        D.summary(tok, model, rows[:a.n], prompt_mode=a.prompt, chat=chat)


if __name__ == "__main__":
    main()
