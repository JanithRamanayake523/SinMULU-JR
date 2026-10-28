"""Score answer files on the 300-question test subset with its answer key (2026-10-27). Evaluation only.

  python -m sinmmlu.score_subset [--key dev/test_subset300_with_answers.jsonl] [--runs submissions/test_subset300]

Reads every <runs>/<exp_id>/predictions.zip (and the full test submissions in --full), keeps the 300 subset questions,
and prints accuracy overall, per level and as the macro average of the three levels (the shared-task metric), with the
gain over the base model under the same prompt and the exact McNemar p-value. Writes <runs>/scores.csv.
The answer key is never used for training or prompt design.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

from .data import LEVELS, ROOT, load_jsonl, path, submission_key
from .metrics import mcnemar_exact

MODELS = {"gemma3_4b_it": "base", "gemma3_4b_it_lora_gmmlu": "A", "gemma3_4b_it_lora_sinmulu": "B1",
          "gemma3_4b_it_lora_sinmulu_ep2": "B2", "gemma3_4b_it_lora_sinmulu_r8": "C1",
          "gemma3_4b_it_lora_sinmulu_r8_ep2": "C2"}
FULL = {"submission 2 (full file)": "submissions/FINAL/predictions.jsonl",
        "A (full file)": "submissions/2026-10-27_lora-global-mmlu-si_cot_en_role_mt1024/predictions.jsonl",
        "B2 (full file)": "submissions/2026-10-27_lora-sinmulu-si-ep2_cot_en_role_mt1024/predictions.jsonl",
        "qwen base (full file)": "submissions/2026-10-27_qwen35-4b_cot_en_role_mt1024/predictions.jsonl",
        "qwen combined LoRA (full file)": "submissions/2026-10-28_qwen35-4b_lora-combined-si_cot_en_role_mt1024/predictions.jsonl",
        "qwen combined LoRA think (full file)": "submissions/2026-10-28_qwen35-4b_lora-combined-si_cot_en_role_think_mt4096/predictions.jsonl"}
PROMPTS = ["paper_nosubject", "paper_subject_si", "en_instr_nosubject", "role_subject_si", "cot_si", "cot_en",
           "cot_en_role"]


def read(lines, keys):
    p = {submission_key(r): int(r["answer"]) for r in map(json.loads, filter(str.strip, lines))}
    return {k: p[k] for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="dev/test_subset300_with_answers.jsonl")
    ap.add_argument("--runs", default="submissions/test_subset300")
    a = ap.parse_args()
    gold = {submission_key(r): int(r["answer"]) for r in load_jsonl(a.key)}
    runs = {}
    for d in sorted(path(a.runs).iterdir()):
        z = d / "predictions.zip"
        if z.exists():
            parts = d.name.split("__")
            runs[(MODELS.get(parts[0], parts[0]), parts[2])] = read(
                zipfile.ZipFile(z).read("predictions.jsonl").decode("utf-8").splitlines(), gold)
    for name, f in FULL.items():
        if path(f).exists():
            runs[(name, "cot_en_role")] = read(path(f).read_text(encoding="utf-8").splitlines(), gold)

    def acc(p, lvl=None):
        ks = [k for k in gold if lvl is None or k[0] == lvl]
        return 100 * sum(p[k] == gold[k] for k in ks) / len(ks)

    rows = []
    order = [(m, pr) for pr in PROMPTS for m in ("base", "A", "B1", "B2", "C1", "C2") if (m, pr) in runs]
    order += [k for k in runs if k not in order]
    print(f"{'model':26s} {'prompt':20s} {'all':>5s} {'easy':>5s} {'med':>5s} {'hard':>5s} {'MACRO':>6s} "
          f"{'gain':>6s} {'p':>6s}")
    for key in order:
        p, (m, pr) = runs[key], key
        lv = [acc(p, l) for l in LEVELS]
        b = runs.get(("base", pr))
        gain = pv = ""
        if m != "base" and b is not None:
            pv = mcnemar_exact([p[k] == gold[k] for k in gold], [b[k] == gold[k] for k in gold])
            gain, pv = f"{acc(p) - acc(b):+.1f}", f"{pv:.3f}"
        print(f"{m:26s} {pr:20s} {acc(p):5.1f} {lv[0]:5.1f} {lv[1]:5.1f} {lv[2]:5.1f} {sum(lv) / 3:6.1f} {gain:>6s} {pv:>6s}")
        rows.append({"model": m, "prompt": pr, "all": round(acc(p), 2), "easy": round(lv[0], 2),
                     "medium": round(lv[1], 2), "hard": round(lv[2], 2), "macro": round(sum(lv) / 3, 2),
                     "gain_vs_base": gain, "p": pv})
    with open(path(a.runs) / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(gold)} questions; noise: about +-5.7 points on all 300, +-10 per level")


if __name__ == "__main__":
    main()
