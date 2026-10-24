"""Run the selected system on the public test set and produce the Codabench file.

  python -m sinmmlu.predict_winner [--winner results/winner_inference_rest.json]
Reads the winner written by sinmmlu.select_winner, runs its condition(s) with sinmmlu.evaluate on --data test (an
ensemble: every part, then sinmmlu.ensemble), and prints where the validated predictions.zip is. Uploading to
Codabench is a manual step.
"""
import argparse
import json
import subprocess
import sys

from .data import ROOT, path
from .evaluate import exp_id, load_model_cfg


def evaluate(model_cfg, cond):
    cmd = [sys.executable, "-m", "sinmmlu.evaluate", "--model", model_cfg, "--data", "test",
           "--prompt", cond["prompt"], "--chat", cond["chat"], "--perms", str(cond["perms"]),
           "--calibrate", str(cond["calibrate"]), "--translator", cond["translator"]]
    if cond.get("max_new_tokens"):
        cmd += ["--max-new-tokens", str(cond["max_new_tokens"])]
    if cond.get("samples", 1) > 1:
        cmd += ["--samples", str(cond["samples"]), "--temperature", str(cond.get("temperature", 0.7))]
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", default="results/winner_inference_rest.json")
    a = ap.parse_args()
    w = json.loads(path(a.winner).read_text(encoding="utf-8"))
    model_cfg = f"configs/models/{w['model_cfg']}.yaml"
    mcfg = load_model_cfg(model_cfg)
    recipe = w["recipe"]
    print(f"winner on {w['data']}: {w['winner']}", flush=True)
    if "ensemble" in recipe:
        for cond in recipe["ensemble"]:
            evaluate(model_cfg, cond)
        parts = [exp_id(mcfg, c, "test") for c in recipe["ensemble"]]
        subprocess.run([sys.executable, "-m", "sinmmlu.ensemble", "--name", recipe["name"], "--exps", *parts],
                       check=True, cwd=ROOT)
        out = ROOT / "outputs" / f"{recipe['name']}__test"
    else:
        evaluate(model_cfg, recipe["condition"])
        out = ROOT / "outputs" / exp_id(mcfg, recipe["condition"], "test")
    z = out / "predictions.zip"
    if not z.exists():
        raise SystemExit(f"{z} missing: the submission failed validation (see above)")
    print(f"SUBMISSION READY: {z}")


if __name__ == "__main__":
    main()
