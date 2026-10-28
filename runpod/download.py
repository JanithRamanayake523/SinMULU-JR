"""Download the models of the given waves into $HF_HOME (fail fast on gated models before a long run).

  python runpod/download.py final                # the final system: Qwen/Qwen3.5-4B (pinned revision, not gated)
  python runpod/download.py gemma                # submissions 1-2 and the Gemma LoRA runs: gemma-3-4b-it
  python runpod/download.py wave1 wave2 wave3    # all models of the screening study
Gated models (Gemma) need HF_TOKEN in the environment (read by huggingface_hub; never written to disk by this script)
and the licence accepted on huggingface.co for that account.
"""
import sys

from huggingface_hub import snapshot_download

WAVES = {
    "final": ["Qwen/Qwen3.5-4B"],  # the final system (+ our LoRA adapter, lora_adapters/ or the HF submission repo)
    "gemma": ["google/gemma-3-4b-it"],
    "wave1": ["google/gemma-3-4b-it", "google/gemma-3-4b-pt", "google/gemma-3-1b-it"],
    "wave2": ["facebook/nllb-200-distilled-600M"],
    "wave3": ["google/madlad400-3b-mt"],
}
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*", "spiece*", "sentencepiece*"]
BIN_ONLY = {"facebook/nllb-200-distilled-600M"}  # may only have pytorch_model.bin
REVISIONS = {"Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}

if __name__ == "__main__":
    waves = sys.argv[1:] or list(WAVES)
    for w in waves:
        for repo in WAVES[w]:
            pats = PATTERNS + (["*.bin"] if repo in BIN_ONLY else [])
            try:
                p = snapshot_download(repo, revision=REVISIONS.get(repo), allow_patterns=pats)
                print(f"ok   {repo} -> {p}", flush=True)
            except Exception as e:
                sys.exit(f"FAILED {repo}: {type(e).__name__}: {e}\n(gated? accept the licence on huggingface.co/{repo})")
