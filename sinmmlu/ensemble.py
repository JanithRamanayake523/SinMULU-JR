"""Average the option probabilities of finished experiments (same model, same data) into a new experiment.
No model is run; the result is saved like any other experiment (dev: results row; test: predictions.zip).

  python -m sinmmlu.ensemble --name ens_b1_b3_c1 --exps <exp_id> <exp_id> [<exp_id> ...]
The experiment id becomes <name>__<data tag>. All parts must use the same base model (the 8B limit counts each model
once, so prompts of one model can be combined freely).
"""
import argparse
import json

from .data import ROOT, load_jsonl, load_rows, path
from .evaluate import save_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--exps", nargs="+", required=True)
    a = ap.parse_args()
    cfgs = [json.loads((ROOT / "outputs" / e / "config.json").read_text(encoding="utf-8")) for e in a.exps]
    data = {c["data"] for c in cfgs}
    models = {c.get("model_cfg", {}).get("model") for c in cfgs}
    assert len(data) == 1, f"experiments use different data: {data}"
    assert len(models) == 1, f"experiments use different models: {models}"
    data = data.pop()
    rows = load_rows(data)
    preds = [{p["uid"]: p["probs"] for p in load_jsonl(ROOT / "outputs" / e / "preds.jsonl")} for e in a.exps]
    for e, pr in zip(a.exps, preds):
        assert set(pr) == {r["uid"] for r in rows}, f"{e} does not cover {data}"
    probs = [[sum(pr[r["uid"]][j] for pr in preds) / len(preds) for j in range(4)] for r in rows]
    tag = "test" if data == "test" else path(data).stem
    save_experiment(f"{a.name}__{tag}", data, rows, probs, {},
                    {"method": "ensemble", "parts": a.exps, "model_cfg": cfgs[0].get("model_cfg", {}),
                     "env": cfgs[0].get("env", {})})


if __name__ == "__main__":
    main()
