"""Rebuild the merged model of a LoRA run from the base model + its saved adapter (2026-10-27).

  python -m sinmmlu.merge_lora --model configs/models/gemma3_4b_it.yaml \\
      --adapter lora_adapters/gemma3-4b-it_lora_global-mmlu-si_epoch1/adapter --out models/gemma3_4b_it_lora_gmmlu

Does what the end of sinmmlu.train_lora does (base loaded by model.load_lm in bf16, adapter in fp32, merge_and_unload,
bf16 safetensors + tokenizer), so the result is the same checkpoint without retraining.
"""
import argparse
import json

from .data import path
from .evaluate import load_model_cfg
from .model import load_lm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model config, configs/models/<name>.yaml")
    ap.add_argument("--adapter", required=True, help="folder with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from peft import PeftModel
    mcfg = load_model_cfg(a.model)
    model, tok, info = load_lm(mcfg["model"], mcfg["revision"], mcfg["dtype"], "none")
    print(f"[model] {mcfg['model']} {json.dumps(info)}", flush=True)
    model = PeftModel.from_pretrained(model, str(path(a.adapter))).merge_and_unload()
    out = path(a.out)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[done] merged -> {out}", flush=True)


if __name__ == "__main__":
    main()
