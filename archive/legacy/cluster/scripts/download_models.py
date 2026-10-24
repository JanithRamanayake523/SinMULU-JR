"""Download the models used by configs/*.yaml (and any extra) into the Hugging Face cache, so jobs can run offline.
Run on a node WITH internet (usually the login node), with HF_TOKEN set for the gated Gemma / Llama models:
    export HF_TOKEN=hf_xxx          # your own token; accept the licences on the model pages first
    python scripts/download_models.py
    python scripts/download_models.py --extra google/gemma-3-1b-pt
"""
import argparse, glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))


def main():
    import yaml
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="additional model ids")
    ap.add_argument("--all", action="store_true", help="every config in configs/ (default: only the three 3-4B models)")
    ap.add_argument("--only", nargs="*", default=None, help="config names to download, e.g. --only gemma3_4b")
    a = ap.parse_args()
    default = set(a.only) if a.only else {"gemma3_4b", "llama32_3b", "qwen3_4b_instruct_2507"}
    ids = []
    for f in sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))):
        if not a.all and Path(f).stem not in default:
            continue
        m = yaml.safe_load(open(f, encoding="utf-8")).get("model")
        if m and m not in ids:
            ids.append(m)
    ids += [m for m in a.extra if m not in ids]
    print("HF_HOME =", os.environ["HF_HOME"])
    for m in ids:
        try:
            path = snapshot_download(m, token=os.environ.get("HF_TOKEN"), allow_patterns=[
                "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*", "*.py"])
            print("OK  ", m, "->", path)
        except Exception as e:  # gated model without access, no internet, ...
            print("FAIL", m, type(e).__name__, str(e)[:200])
            sys.exit(1)
    print("done. Use OFFLINE=1 in the job scripts so compute nodes do not need internet.")


if __name__ == "__main__":
    main()
