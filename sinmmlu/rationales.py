"""Worked examples for reasoning prompts: the model's own English reasoning on the few-shot pool questions, kept only
when its stated answer is the gold answer (dev data only; the pool is disjoint from every evaluation set).

  python -m sinmmlu.rationales --model configs/models/gemma3_4b_it.yaml
Writes dev/fewshot_cot_en.jsonl (pool rows + "rationale", the reasoning up to and including "Answer: N"), in the pool
order, so the first examples of a subject are used first.
"""
import argparse

from . import prompts as P
from .data import load_jsonl, write_jsonl
from .evaluate import ANSWER_RE, load_model_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="cot_en")
    a = ap.parse_args()
    from .model import LM, load_lm, seed_everything
    cfg = load_model_cfg(a.model)
    seed_everything(cfg["seed"])
    model, tok, info = load_lm(cfg["model"], cfg["revision"], cfg["dtype"], cfg["quant"])
    lm = LM(model, tok, chat=True, context=info["context_length"], gen_batch_size=cfg["gen_batch_size"])
    pool = load_jsonl(P.FEWSHOT_POOL)
    p = P.PROMPTS[a.prompt]
    gens = lm.generate([P.build(r, a.prompt) for r in pool], p.max_new_tokens)
    kept = []
    for r, g in zip(pool, gens):
        m = list(ANSWER_RE.finditer(g))
        if m and int(m[-1].group(1)) == int(r["answer"]):
            kept.append(dict(r, rationale=g[:m[-1].end()].strip()))
    write_jsonl(P.RATIONALE_POOL, kept)
    subjects = {r["file"] for r in pool}
    covered = {r["file"] for r in kept}
    print(f"{len(kept)}/{len(pool)} pool questions with a correct rationale; subjects without one: "
          f"{sorted(subjects - covered) or 'none'} -> {P.RATIONALE_POOL}")


if __name__ == "__main__":
    main()
