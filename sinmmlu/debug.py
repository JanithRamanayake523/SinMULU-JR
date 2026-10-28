"""What does the model want to output after the prompt? DEV DATA ONLY (never the test set).

  python -m sinmmlu.debug --model configs/models/gemma3_4b_it.yaml --n 5 [--prompt paper_subject_si] [--chat off] [--generate 60]
Per question: the tail of the exact scored text, the top-k next tokens, the digit probabilities and mass, the gold
answer, and optionally a greedy continuation.
"""
import argparse

from . import prompts as P
from .data import load_jsonl
from .evaluate import load_model_cfg
from .model import LM, label_token_ids, load_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="dev/archive/easy_full.jsonl")
    ap.add_argument("--prompt", default="paper_nosubject", choices=sorted(P.PROMPTS))
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--generate", type=int, default=0, help="greedy continuation tokens (0 = off)")
    ap.add_argument("--suffix", default="", help="text appended after the rendered prompt")
    ap.add_argument("--no-prefill", action="store_true", help="score without the answer pre-fill (as before 2026-10-24)")
    ap.add_argument("--quiet", action="store_true", help="only the per-question label mass and a summary")
    a = ap.parse_args()
    assert a.data != "test", "debug output is for labelled dev data only"
    import torch
    cfg = load_model_cfg(a.model)
    chat = cfg["chat_capable"] if a.chat == "auto" else a.chat == "on"
    model, tok, info = load_lm(cfg["model"], cfg["revision"], cfg["dtype"], cfg["quant"])
    lm = LM(model, tok, chat=chat, batch_size=1, context=info["context_length"])
    labels = label_token_ids(tok, P.PROMPTS[a.prompt].labels)
    dev = next(model.parameters()).device
    print("label token ids:", dict(zip(P.PROMPTS[a.prompt].labels, labels)))
    masses, correct = [], 0
    for r in load_jsonl(a.data)[a.start:a.start + a.n]:
        word = None if a.no_prefill else P.PROMPTS[a.prompt].answer_word
        text = lm.render(P.build(r, a.prompt), word) + a.suffix
        ids = lm.encode([text])[0]
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        top = torch.topk(lp, a.topk)
        digit = [torch.logsumexp(lp[v], 0).exp().item() for v in labels]
        masses.append(sum(digit))
        correct += max(range(4), key=lambda j: digit[j]) + 1 == int(r["answer"])
        if a.quiet:
            continue
        print(f"\n=== {r['uid']} gold={r['answer']} tokens={len(ids)}")
        print("text tail:", repr(text[-160:]))
        print("last tokens:", [tok.decode([i]) for i in ids[-4:]])
        print("top next tokens:", [(tok.decode([i]), round(v.exp().item(), 3)) for v, i in zip(top.values, top.indices.tolist())])
        print("labels:", [round(p, 3) for p in digit], "mass:", round(sum(digit), 3))
        if a.generate:
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=a.generate, do_sample=False)
            print("greedy:", repr(tok.decode(g[0, len(ids):], skip_special_tokens=True)))
    print(f"SUMMARY prompt={a.prompt} chat={chat} suffix={a.suffix!r} n={len(masses)} "
          f"mean_label_mass={sum(masses) / len(masses):.3f} min={min(masses):.3f} acc={correct / len(masses):.3f}")

if __name__ == "__main__":
    main()
