"""Download the models of the given waves into $HF_HOME (fail fast on gated models before a long run).

  python runpod/download.py wave1 wave2 wave3
Needs HF_TOKEN in the environment (read by huggingface_hub; never written to disk by this script) and the Gemma
licence accepted on huggingface.co for that account.
"""
import sys

from huggingface_hub import snapshot_download

WAVES = {
    "wave1": ["google/gemma-3-4b-it", "google/gemma-3-4b-pt", "google/gemma-3-1b-it"],
    "wave2": ["facebook/nllb-200-distilled-600M"],
    "wave3": ["google/madlad400-3b-mt"],
}
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*", "spiece*", "sentencepiece*"]
BIN_ONLY = {"facebook/nllb-200-distilled-600M"}  # may only have pytorch_model.bin

if __name__ == "__main__":
    waves = sys.argv[1:] or list(WAVES)
    for w in waves:
        for repo in WAVES[w]:
            pats = PATTERNS + (["*.bin"] if repo in BIN_ONLY else [])
            try:
                p = snapshot_download(repo, allow_patterns=pats)
                print(f"ok   {repo} -> {p}", flush=True)
            except Exception as e:
                sys.exit(f"FAILED {repo}: {type(e).__name__}: {e}\n(gated? accept the licence on huggingface.co/{repo})")
