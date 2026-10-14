"""Offline test of permutation averaging + calibration + output format using an oracle scorer (no model needed).
Run: python -X utf8 tests/test_eval_core.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import eval_logprob as E
from data_utils import cyclic_perms, load_jsonl, permute_options

rows = load_jsonl("dev/dev_easy_screen.jsonl")[:40]
cfg = dict(E.DEFAULTS, perms=4, prompt="subject_si")
oracle = {}
for r in rows:
    for perm in cyclic_perms(4):
        pr = permute_options(r, perm)
        lp = [-10.0] * 4
        lp[pr["answer"] - 1] = -0.1
        oracle[E.prompt_for(pr, cfg["prompt"])] = lp
scorer = lambda ps: [oracle.get(p, [-1.0] * 4) for p in ps]
probs = E.score_rows(rows, scorer, cfg)
assert all(abs(sum(p) - 1) < 1e-6 for p in probs)
assert all(max(range(4), key=lambda j: p[j]) + 1 == r["answer"] for r, p in zip(rows, probs)), "perm mapping broken"
# position-biased model (always loves digit 1): permutation averaging must spread predictions
biased = lambda ps: [[0.0, -2.0, -2.0, -2.0] for _ in ps]
pb = E.score_rows(rows, biased, dict(cfg, perms=4))
assert all(abs(p[0] - p[1]) < 1e-9 for p in pb), "averaging should remove pure position bias"
# calibration removes a constant positional prior
pc = E.score_rows(rows, biased, dict(cfg, perms=1, calibrate=True))
assert all(abs(p[0] - 0.25) < 1e-9 for p in pc), "calibration should neutralise a pure positional prior"
print("OK: oracle accuracy 100%, bias removal OK, calibration OK")

# ---- regression tests for the supervisor's 16:35 review (B1, B2)
import argparse, yaml
assert yaml.safe_load("chat: on")["chat"] is True, "documenting the YAML 1.1 trap"
ns = argparse.Namespace(config="configs/qwen3_0.6b.yaml", data_tag="dev", limit=None, **{k: None for k in E.DEFAULTS})
for raw, want in ((True, "on"), (False, "off"), ("on", "on"), ("auto", "auto")):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp()); f = d / "c.yaml"
    f.write_text(f"model: x/y\nchat: {raw if isinstance(raw, bool) else repr(raw)}\n", encoding="utf-8")
    # yaml 'True'/'False' scalars parse to bools just like on/off do
    ns.config = str(f)
    cfg = E.load_cfg(ns)
    assert cfg["chat"] == want, (raw, cfg["chat"])
f.write_text("model: x/y\nchat: maybe\n", encoding="utf-8")
try:
    E.load_cfg(ns); raise SystemExit("chat: maybe should be rejected")
except AssertionError:
    pass
assert E.parse_bool("false") is False and E.parse_bool("true") is True and E.parse_bool(None) is None
assert E.check_length([[1] * 10, [1] * 50], 100) == 50
try:
    E.check_length([[1] * 3000], 1024); raise SystemExit("long prompt must raise")
except ValueError:
    pass
print("OK: YAML on/off -> resolved chat, bad chat rejected, --chat-capable bool parsing, long prompt raises")
