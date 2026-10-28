"""LoRA fine-tuning of the answer digit (2026-10-27).

  python -m sinmmlu.train_lora --model configs/models/gemma3_4b_it.yaml \\
      --train data/global_mmlu_si/competition_format/global_mmlu_si.json --out models/gemma3_4b_it_lora_gmmlu
  (smoke test: add --limit 400 --max-steps 10 --out models/smoke)

What is trained: log P(gold digit | chat prompt pre-filled with "<answer_word>: "), exactly the quantity the digit
scorer reads (model.LM._forward: left padding, position ids from the mask, last position only). The loss is on that
one token only: no loss on the question, the pre-fill or an end-of-turn token, so the model is not taught to stop
after a digit (the reasoning prompt of submission 2 has to keep working). The training prompt (default
en_instr_nosubject, a plain digit prompt) is deliberately not the reasoning prompt used for evaluation.

Batches: each optimizer step takes `--batch` random examples; inside a step they are sorted by length and cut into
micro-batches of at most `--max-batch-tokens` padded tokens (little padding, no packing across examples).
Validation (scored every --val-every steps): the questions of --val, or else those of --train with q_no >= --val-from
(Global-MMLU dev split), which are then not trained on.

Writes <out>/ : the merged model (bf16 safetensors + tokenizer; load it with a model config whose `model` is <out>),
<out>/adapter/ (LoRA weights only), <out>/train_config.json, <out>/train_log.jsonl.
LoRA on the language model's attention and MLP projections only; the vision tower is untouched. Merging adds no
parameters: the merged checkpoint has exactly as many as the base model.
"""
import argparse
import json
import math
import random
import time

from . import prompts as P
from .data import ROOT, path
from .evaluate import load_model_cfg
from .model import env_info, label_token_ids, load_lm, render, seed_everything

# Language-model layers only: "...language_model..." (Gemma 3, loaded with its vision tower) or "model.layers.N..."
# (Qwen3.5, which model.load_lm loads as the text-only Qwen3_5ForCausalLM). in_proj_qkv / in_proj_z / out_proj are the
# linear-attention (Gated DeltaNet) layers of Qwen3.5 (24 of its 32 layers). Gemma 3's language model has no modules
# with these names and no top-level "model.layers", so Gemma adapters get exactly the same 238 targets as before.
TARGETS = (r"(.*language_model.*|model\.layers\.\d+\..*)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
           r"|in_proj_qkv|in_proj_z|out_proj)")


def plan_steps(lengths, batch, max_tokens, rng):
    """Epoch plan: list of steps, each a list of micro-batches (lists of example indices)."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    steps = []
    for s in range(0, len(order), batch):
        idx = sorted(order[s:s + batch], key=lambda i: lengths[i])
        micro, cur = [], []
        for i in idx:  # sorted ascending, so the newest example is the longest of `cur`
            if cur and (len(cur) + 1) * lengths[i] > max_tokens:
                micro.append(cur)
                cur = []
            cur.append(i)
        micro.append(cur)
        steps.append(micro)
    return steps


def gold_logprob(lp, gold_ids):
    """lp: (batch, vocab) log-probs at the answer position; gold_ids[b]: token ids of example b's gold label."""
    import torch
    return torch.stack([torch.logsumexp(lp[b, ids], dim=0) for b, ids in enumerate(gold_ids)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model config, configs/models/<name>.yaml")
    ap.add_argument("--train", required=True, help="JSON list in the competition format, with answers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="en_instr_nosubject", choices=[n for n, p in P.PROMPTS.items()
                                                                      if p.method == "digits" and p.lang == "si"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of the steps")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32, help="examples per optimizer step")
    ap.add_argument("--max-batch-tokens", type=int, default=12288, help="padded tokens per micro-batch")
    ap.add_argument("--max-len", type=int, default=1024, help="longer prompts are dropped (counted in the config)")
    ap.add_argument("--val", help="validation JSON (same format); overrides --val-from")
    ap.add_argument("--val-from", default="gsi14040", help="q_no >= this is held out ('' = no validation)")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, help="stop after N steps (smoke test)")
    ap.add_argument("--limit", type=int, help="first N training questions only (smoke test)")
    ap.add_argument("--no-merge", action="store_true", help="save the adapter only")
    ap.add_argument("--init-adapter", help="continue training this saved adapter folder (a further epoch); "
                                           "--rank/--alpha/--dropout are then taken from it")
    ap.add_argument("--shuffle-seed", type=int, help="seed of the example order (default: the model config's "
                                                     "seed; give a new one when continuing, e.g. 43)")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    mcfg = load_model_cfg(a.model)
    seed_everything(mcfg["seed"])
    out = path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, minfo = load_lm(mcfg["model"], mcfg["revision"], mcfg["dtype"], "none")
    print(f"[model] {mcfg['model']} {json.dumps(minfo)}", flush=True)

    p = P.PROMPTS[a.prompt]
    lab = label_token_ids(tok, p.labels)
    rows = json.load(open(path(a.train), encoding="utf-8"))
    if a.val:
        val, train = json.load(open(path(a.val), encoding="utf-8")), rows[:a.limit]
    else:
        val = [r for r in rows if a.val_from and str(r["q_no"]) >= a.val_from]
        train = [r for r in rows if not (a.val_from and str(r["q_no"]) >= a.val_from)][:a.limit]

    def encode(rs):
        texts = [render(tok, P.build(r, a.prompt), True, p.answer_word) for r in rs]
        ids = tok(texts, add_special_tokens=False)["input_ids"]  # the chat template already holds BOS
        keep = [k for k, x in enumerate(ids) if len(x) <= a.max_len]
        return [ids[k] for k in keep], [lab[int(rs[k]["answer"]) - 1] for k in keep], len(ids) - len(keep), texts[0]

    tr_ids, tr_gold, tr_long, example = encode(train)
    va_ids, va_gold, va_long, _ = encode(val) if val else ([], [], 0, None)
    lengths = [len(x) for x in tr_ids]
    print(f"[data] train {len(tr_ids)} (dropped {tr_long} longer than {a.max_len} tokens), val {len(va_ids)} | "
          f"tokens: {sum(lengths)} total, mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}", flush=True)
    print(f"[data] example prompt (loss on the next token only):\n{example}", flush=True)

    if a.init_adapter:  # continue a saved adapter (its rank/alpha/targets), e.g. for a further epoch
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(path(a.init_adapter)), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules=TARGETS, bias="none"))
    n_lora = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[lora] trainable parameters: {n_lora:,}", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    dev = next(model.parameters()).device

    def forward(ids):
        enc = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(dev)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=pos, logits_to_keep=1, use_cache=False).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits")
        return torch.log_softmax(logits, dim=-1)

    def correct(lp, gold):
        scores = torch.stack([torch.logsumexp(lp[:, ids], dim=1) for ids in lab], dim=1)
        return [int(scores[b].argmax()) == lab.index(g) for b, g in enumerate(gold)]

    def validate():
        if not va_ids:
            return {}
        model.eval()
        loss, ok = 0.0, 0
        with torch.no_grad():
            for micro in plan_steps([len(x) for x in va_ids], len(va_ids), a.max_batch_tokens, random.Random(0))[0]:
                lp = forward([va_ids[i] for i in micro])
                g = [va_gold[i] for i in micro]
                loss -= gold_logprob(lp, g).sum().item()
                ok += sum(correct(lp, g))
        model.train()
        return {"val_loss": loss / len(va_ids), "val_acc": ok / len(va_ids)}

    rng = random.Random(mcfg["seed"] if a.shuffle_seed is None else a.shuffle_seed)
    plans = [plan_steps(lengths, a.batch, a.max_batch_tokens, rng) for _ in range(a.epochs)]
    total = sum(len(pl) for pl in plans)
    if a.max_steps:
        total = min(total, a.max_steps)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(a.warmup * total), total)
    log = open(out / "train_log.jsonl", "w", encoding="utf-8")

    def record(d):
        log.write(json.dumps(d) + "\n")
        log.flush()
        print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()),
              flush=True)

    record({"step": 0, **validate()})
    model.train()
    t0, step, win_loss, win_ok, win_n, win_tok = time.time(), 0, 0.0, 0, 0, 0
    t_win = t0
    for epoch, plan in enumerate(plans, 1):
        for micro_batches in plan:
            if step >= total:
                break
            n = sum(len(m) for m in micro_batches)
            for micro in micro_batches:
                lp = forward([tr_ids[i] for i in micro])
                g = [tr_gold[i] for i in micro]
                nll = -gold_logprob(lp, g)
                (nll.sum() / n).backward()
                win_loss += nll.sum().item()
                win_ok += sum(correct(lp.detach(), g))
                win_tok += len(micro) * max(len(tr_ids[i]) for i in micro)
            win_n += n
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                el = time.time() - t0
                d = {"step": step, "of": total, "epoch": epoch, "loss": win_loss / win_n, "acc": win_ok / win_n,
                     "lr": sched.get_last_lr()[0], "padded_tok_per_s": win_tok / (time.time() - t_win),
                     "elapsed_min": el / 60, "eta_min": el / step * (total - step) / 60,
                     "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                if a.val_every and (step % a.val_every == 0 or step == total):
                    d.update(validate())
                record(d)
                win_loss, win_ok, win_n, win_tok, t_win = 0.0, 0, 0, 0, time.time()
    train_s = time.time() - t0
    log.close()

    model.save_pretrained(out / "adapter")
    cfg = {"base": mcfg, "base_info": minfo, "args": vars(a), "train_file": a.train, "n_train": len(tr_ids),
           "n_dropped_long": tr_long, "n_val": len(va_ids), "steps": step, "lora_trainable_params": n_lora,
           "target_modules": TARGETS, "train_seconds": round(train_s), "env": env_info(), "merged": not a.no_merge}
    if not a.no_merge:
        model = model.merge_and_unload()
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    print(f"[done] {step} steps in {train_s / 60:.1f} min -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}",
          flush=True)


if __name__ == "__main__":
    main()
