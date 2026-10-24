"""Assemble cluster_package/: everything that has to be copied to the GPU cluster (scripts only, no notebooks).
Rebuild it whenever the code changes:  python tools/make_cluster_package.py
Contents: src/, configs/, tests/, data/ (public train + test), dev/, results/ (results.csv, fertility.csv),
outputs/ (finished experiments listed in results.csv), and the files from cluster/ (README_CLUSTER.md, requirements-cluster.txt, scripts/)."""
import csv, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cluster_package"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def main():
    kept = None
    if (OUT / ".hf_token").exists():  # a token file the user placed here by hand survives rebuilds (its content is never read or printed)
        kept = (OUT / ".hf_token").read_bytes()
    if OUT.exists():
        shutil.rmtree(OUT)  # only this generated folder is removed
    OUT.mkdir()
    if kept is not None:
        (OUT / ".hf_token").write_bytes(kept)
    for d in ("src", "configs", "tests", "dev"):
        shutil.copytree(ROOT / d, OUT / d, ignore=IGNORE)
    shutil.copytree(ROOT / "data", OUT / "data", ignore=IGNORE)
    (OUT / "results").mkdir()
    for f in ("results.csv", "fertility.csv"):
        if (ROOT / "results" / f).exists():
            shutil.copy2(ROOT / "results" / f, OUT / "results" / f)
    n = 0
    res = ROOT / "results" / "results.csv"
    if res.exists():
        (OUT / "outputs").mkdir()
        for r in csv.DictReader(open(res, encoding="utf-8", newline="")):
            src = ROOT / "outputs" / r["exp_id"]
            if src.exists():
                shutil.copytree(src, OUT / "outputs" / r["exp_id"])
                n += 1
    shutil.copytree(ROOT / "cluster" / "scripts", OUT / "scripts")
    shutil.copy2(ROOT / "cluster" / "README_CLUSTER.md", OUT / "README_CLUSTER.md")
    shutil.copy2(ROOT / "cluster" / "requirements-cluster.txt", OUT / "requirements-cluster.txt")
    shutil.copy2(ROOT / "cluster" / ".hf_token.example", OUT / ".hf_token.example")  # template only: a real .hf_token is never packaged
    # shell scripts must have LF line endings to run on Linux
    for f in (OUT / "scripts").glob("*.sh"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{OUT}: {size:.1f} MB, {n} finished experiments included")


if __name__ == "__main__":
    main()
