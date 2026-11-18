"""Parameter counts three ways, to check the 8B budget (run on the pod; no weights are loaded into memory).
  1. checkpoint: sum of tensor sizes in the *.safetensors / *.bin files (what is actually stored)
  2. meta, as counted by sinmmlu.model.count_params before 2026-10-24 (tied matrices may be counted twice)
  3. meta after tie_weights(), unique tensors only
"""
import glob
import json
import os
import struct
import sys

import torch
from accelerate import init_empty_weights
import transformers as T

MODELS = ["google/gemma-3-4b-it", "google/gemma-3-4b-pt", "facebook/nllb-200-distilled-600M", "google/madlad400-3b-mt"]


def safetensors_count(folder):
    total = 0
    for f in glob.glob(os.path.join(folder, "*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k != "__metadata__":
                size = 1
                for d in v["shape"]:
                    size *= d
                total += size
    return total


def bin_count(folder):
    total = 0
    for f in glob.glob(os.path.join(folder, "pytorch_model*.bin")):
        sd = torch.load(f, map_location="meta", weights_only=True)
        total += sum(t.numel() for t in sd.values())
    return total


def meta_counts(mid):
    cfg = T.AutoConfig.from_pretrained(mid)
    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"):
        try:
            with init_empty_weights():
                m = getattr(T, cls).from_config(cfg)
            break
        except (ValueError, KeyError):
            continue
    naive = sum(p.numel() for p in m.parameters())
    m.tie_weights()
    unique = {id(p): p.numel() for p in m.parameters()}
    tied = sum(unique.values())
    vision = sum(p.numel() for n, p in m.named_parameters() if "vision" in n or "multi_modal_projector" in n)
    return cls, naive, tied, vision


for mid in MODELS:
    folder = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--" + mid.replace("/", "--"), "snapshots", "*"))[0]
    ck = safetensors_count(folder) or bin_count(folder)
    cls, naive, tied, vision = meta_counts(mid)
    print(f"{mid}: checkpoint={ck:,} meta_naive={naive:,} meta_tied={tied:,} vision_part={vision:,} ({cls})", flush=True)
sys.exit(0)
