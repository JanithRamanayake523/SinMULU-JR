"""Token-length distribution of the eval prompts, one histogram per model in a 1 x 3 grid. Tokenizer only: no model weights,
no GPU, no predictions. Run it BEFORE the grid to see how long the prompts are for each tokenizer.

Prompt = exactly what is fed to the model: Appendix-B prompt (Sinhala subject name) wrapped in the chat template when the
model is chat capable (configs: chat_capable). Eval set = dev/dev_easy_full.jsonl (public train, Easy only).
Output: results/figures/token_hist_<data>.png  (+ summary stats printed)
Usage: python src/token_hist.py [--data dev/dev_easy_full.jsonl] [--configs configs/a.yaml configs/b.yaml configs/c.yaml]"""
import argparse, csv, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import eval_logprob as E
from data_utils import ROOT

DEFAULT_CONFIGS = ["configs/qwen3_0.6b.yaml", "configs/gemma3_1b.yaml", "configs/llama32_1b.yaml"]


def token_lengths(cfg, rows):
    """Prompt lengths in tokens (as scored), the model context length, and tokeniser fertility."""
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    mc = AutoConfig.from_pretrained(cfg["model"], revision=cfg["revision"])
    ctx = getattr(mc, "max_position_embeddings", None) or getattr(getattr(mc, "text_config", None), "max_position_embeddings", None)
    chat = bool(cfg["chat_capable"]) and cfg["chat"] != "off"
    lens = []
    for r in rows:
        p = E.prompt_for(r, cfg["prompt"])
        if chat:
            p = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, date_string=E.FIXED_DATE)
        lens.append(len(tok(p, add_special_tokens=not chat)["input_ids"]))
    fert = E.fertility(tok, rows)  # tokens per Sinhala word on the question text (all eval questions)
    return lens, ctx, chat, fert


def plot(results, out, data_label):
    """results: list of (title, lengths, ctx). Shared x/y axes so the three tokenizers are directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    top = max(max(l) for _, l, _ in results)
    bins = np.linspace(0, top * 1.02, 41)
    for ax, (title, lens, ctx) in zip(axes, results):
        a = np.array(lens)
        ax.hist(a, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
        for val, style, lab in ((np.median(a), "-", "median"), (np.percentile(a, 99), "--", "p99"), (a.max(), ":", "max")):
            ax.axvline(val, color="#E45756", linestyle=style, linewidth=1.3, label=f"{lab} {val:.0f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("prompt length (tokens)")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("number of questions")
    fig.suptitle(f"Prompt token-length distribution ({data_label}, n={len(results[0][1])})", fontsize=12)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dev/dev_easy_full.jsonl")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    a = ap.parse_args()
    rows, is_test = E.load_eval_rows(a.data)
    assert not is_test, "histograms are for the labelled eval set; the public test set is not analysed"
    results, ferts = [], []
    for c in a.configs:
        import argparse as ap2
        cfg = E.load_cfg(ap2.Namespace(config=c, data_tag=Path(a.data).stem, limit=None, **{k: None for k in E.DEFAULTS}))
        lens, ctx, chat, fert = token_lengths(cfg, rows)
        s = sorted(lens)
        print(f"{cfg['model']} (chat template {'on' if chat else 'off'}): n={len(s)} min={s[0]} median={s[len(s) // 2]} "
              f"p99={s[int(.99 * len(s))]} max={s[-1]} model_context={ctx} over_context={sum(l > ctx for l in s) if ctx else '?'} "
              f"FERTILITY={fert:.3f} tokens/word")
        ferts.append((cfg["model"], round(fert, 3), len(rows)))
        results.append((f"{cfg['model'].split('/')[-1]}  (context {ctx // 1024 if ctx else '?'}k)", lens, ctx))
    out = ROOT / "results" / "figures" / f"token_hist_{Path(a.data).stem}.png"
    plot(results, out, Path(a.data).name)
    fpath = ROOT / "results" / "fertility.csv"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "fertility_tokens_per_word", "n_questions"])
        w.writerows(ferts)
    print("fertility ->", fpath)
    print("saved", out)


if __name__ == "__main__":
    main()
