"""Offline tests (no model weights, no GPU). Run: python -m pytest -q"""
import random

import pytest
import yaml

from sinmmlu import prompts as P
from sinmmlu.data import cyclic_perms, drop_random_wrong_option, load_jsonl, norm_model, path, permute_options, subject_family
from sinmmlu.evaluate import check_condition, exp_id, parse_answer, parse_bool
from sinmmlu.metrics import holm, mcnemar_exact, paired_bootstrap
from sinmmlu.scoring import argmax1, score_rows
from sinmmlu.translate import join_segments, split_segments

ROWS = load_jsonl("dev/inference_subset.jsonl")[:40]
EN = [dict(r, question_en=f"Q{i}?", choices_en=[f"opt{i}{k}" for k in "abcd"]) for i, r in enumerate(ROWS)]
DIGIT_PROMPTS = [n for n, p in P.PROMPTS.items() if p.method == "digits"]
MCFG = {"name": "m", "model": "x/m", "quant": "none", "chat_capable": True}


def oracle_scorer(rows, prompt):
    table = {}
    for r in rows:
        for perm in cyclic_perms(4):
            pr = permute_options(r, perm)
            lp = [-10.0] * 4
            lp[pr["answer"] - 1] = -0.1
            table[P.build(pr, prompt)] = lp
    return lambda ps, labels, word: [table.get(p, [-1.0] * 4) for p in ps]


@pytest.mark.parametrize("prompt", DIGIT_PROMPTS)
def test_permutation_mapping_recovers_oracle(prompt):
    probs, stats = score_rows(EN, oracle_scorer(EN, prompt), prompt, perms=4)
    assert all(abs(sum(p) - 1) < 1e-9 for p in probs)
    assert [argmax1(p) for p in probs] == [r["answer"] for r in EN]
    assert 0 < stats["label_mass_min"] <= stats["label_mass_mean"] <= 4


@pytest.mark.parametrize("prompt", ["paper_nosubject", "paper_subject_en", "en_nosubject"])
def test_position_bias_removed_by_averaging_and_calibration(prompt):
    biased = lambda ps, labels, word: [[0.0, -2.0, -2.0, -2.0] for _ in ps]
    pb, _ = score_rows(EN, biased, prompt, perms=4)
    assert all(abs(p[0] - p[1]) < 1e-9 for p in pb)
    pc, _ = score_rows(EN, biased, prompt, perms=1, calibrate=True)
    assert all(abs(p[0] - 0.25) < 1e-9 for p in pc)


def test_calibration_inputs_contain_no_question_text():
    seen = []
    scorer = lambda ps, labels, word: seen.extend(ps) or [[0.0] * 4 for _ in ps]
    score_rows(EN[:3], scorer, "bilingual_nosubject", calibrate=True)
    calib = seen[:3]
    assert all(EN[0]["question_en"] not in t and norm_model(EN[0]["question"]) not in t for turns in calib for _, t in turns)


def test_prompts_build():
    r = EN[0]
    for name, p in P.PROMPTS.items():
        if p.shots and not path(p.pool).exists():
            continue  # rationale pool exists only after python -m sinmmlu.rationales (GPU)
        turns = P.build(r, name)
        assert turns[-1][0] == "user" and "‌" not in turns[-1][1]
        assert len(turns) == 2 * p.shots + 1
        if p.method == "digits":
            assert turns[-1][1].endswith(("පිළිතුර:", "Answer:"))
            assert f"\n{p.labels[0]}. " in turns[-1][1] and f"\n{p.labels[3]}. " in turns[-1][1]
    assert r["subject_si"] in P.build(r, "paper_subject_si")[0][1]
    assert "‍" in P.build(r, "paper_nosubject")[0][1]  # ZWJ kept (ප්‍ර)
    assert "opt0a" in P.build(r, "si_q_en_opts")[0][1] and "Q0?" not in P.build(r, "si_q_en_opts")[0][1]
    assert "Q0?" in P.build(r, "bilingual_nosubject")[0][1]
    assert "\n1. " not in P.build(r, "cloze_nosubject")[0][1]  # cloze shows no options


def test_fewshot_examples():
    pool = {e["uid"] for e in load_jsonl(P.FEWSHOT_POOL)}
    subset = {r["uid"] for r in load_jsonl("dev/inference_subset.jsonl")}
    assert not pool & subset
    r = ROWS[0]
    ex = P.examples(r, 5)
    assert len(ex) == 5 and all(subject_family(e) == subject_family(r) for e in ex)
    assert len({e["answer"] for e in ex[:4]}) == 4
    turns = P.build(r, "fewshot3_nosubject")
    assert [t for t, _ in turns] == ["user", "assistant"] * 3 + ["user"]
    assert turns[1][1] == str(ex[0]["answer"])
    # a test subject without training examples falls back to the fixed generic set
    unseen = dict(r, subject="Biology", uid="x")
    assert len(P.examples(unseen, 5)) == 5


def test_subject_family_matches_train_and_test_spellings():
    f = lambda s: subject_family({"subject": s})
    assert f("Arts") == f("Art") and f("Health and Physical Science") == f("Health")
    assert f("Sinhala language and literature") == f("Sinhala") and f("Eastern_Music") == f("Eastern music")


def test_norm_model_keeps_zwj_drops_zwnj():
    assert norm_model("ප්‍ර‌  x \n\n\n y") == "ප්‍ර x\n\ny"


def test_permute_and_drop_option():
    r = dict(EN[0], answer=2)
    pr = permute_options(r, [1, 2, 3, 0])
    assert pr["answer"] == 1 and pr["choices_en"][0] == r["choices_en"][1]
    five = dict(r, choices=list("abcde"))
    d = drop_random_wrong_option(five, random.Random(0))
    assert len(d["choices"]) == 4 and d["choices"][d["answer"] - 1] == "b"


def test_metrics():
    a = [True] * 10 + [False] * 10
    assert mcnemar_exact(a, a) == 1.0
    assert mcnemar_exact([False] * 10, [True] * 10) == pytest.approx(2 / 2 ** 10)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert paired_bootstrap(a, a) == (0, 0)


def test_yaml_and_bool_parsing():
    assert yaml.safe_load("chat: on")["chat"] is True  # the YAML 1.1 trap: chat modes are CLI choices, not YAML
    assert parse_bool("false") is False and parse_bool("True") is True
    with pytest.raises(Exception):
        parse_bool("maybe")


def test_exp_ids_unique_and_conditions_checked():
    ids = {exp_id(MCFG, {"prompt": p, "chat": c, "perms": k, "calibrate": b, "translator": t}, "dev/x.jsonl")
           for p in P.PROMPTS for c in ("on", "off") for k in (1, 4) for b in (False, True) for t in ("none", "nllb")}
    assert len(ids) == len(P.PROMPTS) * 16
    ok = lambda **kw: check_condition({"prompt": "paper_nosubject", "chat": "on", "perms": 1, "calibrate": False,
                                       "translator": "none", **kw}, MCFG)
    ok()
    ok(prompt="en_nosubject", translator="nllb")
    for bad in (dict(prompt="en_nosubject"), dict(translator="nllb"), dict(prompt="cot_si", calibrate=True), dict(prompt="cloze_nosubject", perms=4),
                dict(prompt="cot_si", chat="off"), dict(prompt="cloze_nosubject", calibrate=True)):
        with pytest.raises(SystemExit):
            ok(**bad)


def test_parse_answer():
    assert parse_answer("first... so\nපිළිතුර: 3")[0] == 3
    assert parse_answer("Answer: 2 ... actually **Answer: 4**")[0] == 4
    assert parse_answer("Answer:** 1")[0] == 1
    assert parse_answer("no answer here") == (None, None)
    text = "reasoning. Answer: 2"
    assert text[:parse_answer(text)[1]] == "reasoning. "


def test_segments_roundtrip():
    t = "පළමු වාක්‍යය. දෙවැන්න?\nතෙවැන්න! (i) a\n\nend"
    assert join_segments(split_segments(t)) == t.replace("\n\n", "\n\n")
    assert split_segments("a. b")[0] == ["a.", "b"]


# ----------------------------------------------------------------------------- LM wrapper with a fake model
torch = pytest.importorskip("torch")


class FakeTok:
    """Character tokenizer: id = ord(c) % 97 + 3; 0 = pad, 1 = bos."""
    chat_template = None
    pad_token_id = 0

    def _ids(self, s):
        return [ord(c) % 97 + 3 for c in s]

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [([1] if add_special_tokens else []) + self._ids(t) for t in texts]}

    def encode(self, s, add_special_tokens=True):
        return self(s if isinstance(s, list) else [s], add_special_tokens)["input_ids"][0]

    def pad(self, enc, padding=True, return_tensors="pt"):
        ids = enc["input_ids"]
        n = max(map(len, ids))
        out = {"input_ids": torch.tensor([[0] * (n - len(x)) + x for x in ids]),
               "attention_mask": torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids])}
        return type("Enc", (dict,), {"to": lambda self, d: self})(out)


class FakeModel(torch.nn.Module):
    """Bigram model: logits at position t depend only on token t (so padding cannot change anything)."""

    def __init__(self, vocab=100):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.table = torch.nn.Parameter(torch.randn(vocab, vocab, generator=g))

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
        logits = self.table[input_ids]
        return type("Out", (), {"logits": logits[:, -logits_to_keep:] if logits_to_keep else logits})()


def test_lm_label_scores_and_cloze_match_unbatched_reference():
    from sinmmlu.model import LM
    tok, model = FakeTok(), FakeModel()
    lm = LM(model, tok, chat=False, batch_size=3)
    prompts = [(("user", "ab:"),), (("user", "a much longer prompt:"),), (("user", "x"),), (("user", "yy:"),)]
    lp = lm(prompts, P.DIGITS, "A")
    for p, got in zip(prompts, lp):
        ids = tok.encode(p[0][1] + " ")  # raw pre-fill: a trailing space
        ref = torch.log_softmax(model.table[ids[-1]], -1)
        assert got == pytest.approx([ref[tok.encode(d, False)[0]].item() for d in P.DIGITS], abs=1e-5)
    opts = [["a", "bb", "ccc", "dddd"]] * len(prompts)
    res = lm.cloze(prompts, opts, "A")
    for p, o, got in zip(prompts, opts, res):
        for k, text in enumerate(o):
            seq = tok.encode(p[0][1] + " ") + tok.encode(text, False)
            n = len(tok.encode(text, False))
            ref = sum(torch.log_softmax(model.table[seq[t - 1]], -1)[seq[t]].item() for t in range(len(seq) - n, len(seq)))
            assert got[k][0] == pytest.approx(ref, abs=1e-4) and got[k][1] == n


def test_raw_render_of_fewshot_turns():
    from sinmmlu.model import render
    turns = (("user", "Q1 පිළිතුර:"), ("assistant", "2"), ("user", "Q2 පිළිතුර:"))
    assert render(None, turns, chat=False) == "Q1 පිළිතුර: 2\n\nQ2 පිළිතුර:"
    # label scoring: a trailing space so that the next token is the bare label (Gemma: " 2" = space + "2")
    assert render(None, turns, chat=False, answer_word="පිළිතුර") == "Q1 පිළිතුර: 2\n\nQ2 පිළිතුර: "


def test_every_prompt_marker_matches_its_answer_word():
    r = EN[0]
    for name, p in P.PROMPTS.items():
        if p.method == "digits":
            assert P.build(r, name)[-1][1].endswith(p.answer_word + ":"), name
        if p.method in ("cot", "chain"):
            assert p.cue.endswith(": ") and p.answer_word in p.cue, name


def test_lm_splits_batches_on_out_of_memory():
    from sinmmlu.model import LM

    class Picky(FakeModel):
        def forward(self, input_ids, **kw):
            if input_ids.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("fake")
            return super().forward(input_ids, **kw)

    tok, model = FakeTok(), Picky()
    prompts = [(("user", "q%d:" % i * (i + 1)),) for i in range(7)]
    got = LM(model, tok, chat=False, batch_size=7)(prompts, P.DIGITS, "A")
    ref = LM(FakeModel(), tok, chat=False, batch_size=1)(prompts, P.DIGITS, "A")
    assert [x for r in got for x in r] == pytest.approx([x for r in ref for x in r], abs=1e-6)


def test_rationale_fewshot_turns(tmp_path, monkeypatch):
    pool = [dict(r, rationale=f"Reasoning {i}.\nAnswer: {r['answer']}") for i, r in enumerate(load_jsonl(P.FEWSHOT_POOL))]
    f = tmp_path / "cot.jsonl"
    f.write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in pool), encoding="utf-8")
    monkeypatch.setitem(P.PROMPTS, "cot_test", P.Prompt(P._cot_en, shots=2, pool=str(f),
                                                         example_reply=lambda e: e["rationale"], **P.COT_EN))
    turns = P.build(ROWS[0], "cot_test")
    assert [t for t, _ in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[1][1].startswith("Reasoning") and turns[1][1].endswith(f"Answer: {pool[0]['answer']}")
    assert P.build(dict(ROWS[0], subject="Biology", uid="x"), "cot_test")[1][1].startswith("Reasoning")  # fallback


def test_select_winner_applies_rule(tmp_path, monkeypatch):
    import json, sys
    from sinmmlu import report, select_winner
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(select_winner, "ROOT", tmp_path)
    n = 400
    gold = [1] * n
    def exp(name, n_right, **cfg):
        d = tmp_path / "outputs" / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"exp_id": name, "data": "dev/x.jsonl", **cfg}), encoding="utf-8")
        (d / "preds.jsonl").write_text("".join(json.dumps({"uid": f"u{i}", "pred": 1 if i < n_right else 2, "gold": 1})
                                               + "\n" for i in range(n)), encoding="utf-8")
    cond = {"prompt": "paper_nosubject", "chat": "on", "perms": 1, "calibrate": False, "translator": "none"}
    mc = {"name": "gemma3_4b_it"}
    exp("base", 150, condition=cond, model_cfg=mc)
    exp("small", 154, condition=dict(cond, prompt="paper_subject_si"), model_cfg=mc)   # +1 pt, not significant
    exp("big", 200, condition=dict(cond, prompt="cot_en"), model_cfg=mc)              # +12.5 pt
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(sys, "argv", ["x", "--data", "dev/x.jsonl", "--baseline", "base", "--candidates", "small", "big"])
    select_winner.main()
    w = json.loads((tmp_path / "results" / "winner_x.json").read_text(encoding="utf-8"))
    assert w["winner"] == "big" and w["recipe"]["condition"]["prompt"] == "cot_en"
    assert [r["adopted"] for r in w["table"]] == [False, True]


def test_reasoning_options_in_exp_id_and_checks():
    base = {"prompt": "cot_en_role", "chat": "on", "perms": 1, "calibrate": False, "translator": "none"}
    old = exp_id(MCFG, base, "dev/x.jsonl")
    assert "__mt" not in old and "__sc" not in old  # earlier experiment ids unchanged
    assert exp_id(MCFG, dict(base, max_new_tokens=1024), "dev/x.jsonl").endswith("__trnone__mt1024__x")
    assert "__sc5t0.7" in exp_id(MCFG, dict(base, samples=5, temperature=0.7), "dev/x.jsonl")
    with pytest.raises(SystemExit):
        check_condition(dict(base, prompt="paper_nosubject", max_new_tokens=1024), MCFG)


def test_self_consistency_soft_and_majority_vote(tmp_path):
    from sinmmlu.evaluate import run_cot
    rows = [dict(r) for r in ROWS[:3]]

    class StubLM:
        def __init__(self):
            self.calls = []

        def render(self, turns):
            return turns[-1][1]

        def generate(self, texts, max_new, rendered=False, sample_seed=None, temperature=0.7):
            self.calls.append((max_new, sample_seed))
            s = 0 if sample_seed is None else (sample_seed - 42) // 1000
            # sample 0 and 1 say "2", sample 2 says "3", for every question
            return [f"reasoning {s}\nAnswer: {3 if s == 2 else 2}" for _ in texts]

        def label_logprobs_text(self, texts, labels):
            out = []
            for t in texts:
                d = 3 if "reasoning 2" in t else 2
                out.append([0.0 if j + 1 == d else -5.0 for j in range(4)])
            return out

    lm = StubLM()
    probs, stats = run_cot(lm, rows, "cot_en", tmp_path, {"samples": 3, "temperature": 0.7, "max_new_tokens": 99})
    assert [c[0] for c in lm.calls] == [99, 99, 99] and [c[1] for c in lm.calls] == [42, 1042, 2042]
    assert all(argmax1(p) == 2 for p in probs)
    assert stats["majority_vote_preds"] == [2, 2, 2] and stats["parse_rate"] == 1.0
    assert sorted(f.name for f in tmp_path.glob("generations_s*.jsonl")) == [f"generations_s{i}.jsonl" for i in range(3)]
    run_cot(lm, rows, "cot_en", tmp_path, {"samples": 3, "temperature": 0.7, "max_new_tokens": 99})
    assert len(lm.calls) == 3  # second run reuses the cached reasoning


def test_negation_chain_turns_limits_and_cache(tmp_path):
    from sinmmlu.evaluate import run_chain
    rows = [dict(r) for r in ROWS[:2]]

    class StubLM:
        def __init__(self):
            self.calls = []

        def render(self, turns):
            return " || ".join(f"{role}:{text[:12]}" for role, text in turns)

        def generate(self, texts, max_new, rendered=False, sample_seed=None, temperature=0.7):
            self.calls.append((max_new, texts[0].count("user:")))
            step = len(self.calls) - 1
            replies = ["... Correct options: 2, 3", "... Options for the opposite question: 1, 4", "short.\nAnswer: 3"]
            return [replies[step] for _ in texts]

        def label_logprobs_text(self, texts, labels):
            assert all(t.endswith("\n\nAnswer: ") and "user:Now use both" in t for t in texts)
            return [[-5.0, -5.0, 0.0, -5.0] for _ in texts]

    lm = StubLM()
    probs, stats = run_chain(lm, rows, "cot_en_role_negchain", tmp_path)
    assert lm.calls == [(1024, 1), (1024, 2), (256, 3)]  # limits per step; one more user turn each step
    assert all(argmax1(p) == 3 for p in probs) and stats["parse_rate"] == 1.0
    assert stats["step0_mean_options_listed"] == 2 and stats["step1_mean_options_listed"] == 2
    run_chain(lm, rows, "cot_en_role_negchain", tmp_path)
    assert len(lm.calls) == 3  # all three steps reused from the cache


def test_reasoning_option_order_averaging_maps_back():
    from sinmmlu.evaluate import run_cot
    import tempfile, pathlib
    rows = [dict(r, answer=2) for r in ROWS[:3]]
    gold_text = {id(r): norm_model(r["choices"][1]) for r in rows}

    class StubLM:  # always picks the position that currently shows the gold option text
        def render(self, turns):
            return turns[-1][1]

        def generate(self, texts, max_new, rendered=False, sample_seed=None, temperature=0.7):
            return ["reasoning" for _ in texts]

        def label_logprobs_text(self, texts, labels):
            out = []
            for t in texts:
                pos = next(i for i in range(4) if any(f"\n{i + 1}. {g}" in t for g in gold_text.values()))
                out.append([0.0 if j == pos else -5.0 for j in range(4)])
            return out

    probs, stats = run_cot(StubLM(), rows, "cot_en", pathlib.Path(tempfile.mkdtemp()), {"perms": 2})
    assert all(argmax1(p) == 2 for p in probs) and stats["orders"] == 2
