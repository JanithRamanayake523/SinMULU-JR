"""Tokenizer-only diagnostics (no model weights): fertility and prompt token lengths per model.

  python -m sinmmlu.tokens --models configs/models/gemma3_4b_it.yaml configs/models/qwen3_4b.yaml [--plot results/figures/token_lengths.png]
Fertility = tokens per whitespace-separated word of the question text (lower = the tokenizer knows Sinhala better).
Lengths are measured on the exact scored text (prompt + chat template when the model is chat capable).
"""
import argparse
from pathlib import Path

from . import prompts as P
from .data import load_jsonl, norm_model, path
from .evaluate import load_model_cfg
from .model import context_length, render


def fertility(tok, rows):
    words = sum(len(norm_model(r["question"]).split()) for r in rows)
    toks = sum(len(tok.encode(norm_model(r["question"]), add_special_tokens=False)) for r in rows)
    return toks / max(words, 1)


def prompt_lengths(tok, rows, prompt, chat):
    texts = [render(tok, P.build(r, prompt), chat) for r in rows]
    return [len(x) for x in tok(texts, add_special_tokens=not chat)["input_ids"]]


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(0, max(max(l) for _, l in results) * 1.02, 41)
    for ax, (title, lens) in zip(axes[0], results):
        a = np.array(lens)
        ax.hist(a, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
        for val, style, lab in ((np.median(a), "-", "median"), (np.percentile(a, 99), "--", "p99"), (a.max(), ":", "max")):
            ax.axvline(val, color="#E45756", linestyle=style, linewidth=1.2, label=f"{lab} {val:.0f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("prompt length (tokens)")
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("questions")
    fig.tight_layout()
    Path(path(out)).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path(out), dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="dev/easy_full.jsonl")
    ap.add_argument("--prompt", default="paper_nosubject", choices=sorted(P.PROMPTS))
    ap.add_argument("--plot")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    rows = load_jsonl(a.data)
    results = []
    for m in a.models:
        cfg = load_model_cfg(m)
        tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
        ctx = context_length(AutoConfig.from_pretrained(cfg["model"], revision=cfg["revision"]))
        lens = sorted(prompt_lengths(tok, rows, a.prompt, cfg["chat_capable"]))
        print(f"{cfg['model']}: fertility={fertility(tok, rows):.3f} tokens/word | prompt tokens median={lens[len(lens) // 2]} "
              f"p99={lens[int(.99 * len(lens))]} max={lens[-1]} context={ctx}")
        results.append((cfg["name"], lens))
    if a.plot:
        plot(results, a.plot)


if __name__ == "__main__":
    main()
