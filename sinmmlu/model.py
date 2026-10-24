"""Model loading and the next-token digit scorer.

Precision rules
  dtype auto : bf16 on GPUs that support it (Ampere or newer), otherwise fp32. Never fp16 by default: Gemma 3
               activations overflow in fp16.
  quant 4bit : NF4 (as in the paper) with the compute dtype resolved as above, never fp16.
Every forward pass is checked for non-finite logits; a NaN/inf raises instead of producing silent garbage.
"""
import random
import subprocess

from .data import ROOT

FIXED_DATE = "26 Jul 2024"  # Llama-3 chat templates insert today's date otherwise -> results would depend on the day


def resolve_dtype(name):
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return torch.float32
    return {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[name]


def load_lm(model_id, revision=None, dtype="auto", quant="none"):
    """Returns (model, tokenizer, info). Gemma-3 >= 4B are multimodal checkpoints: if AutoModelForCausalLM refuses them,
    Gemma3ForConditionalGeneration is used with text input only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = resolve_dtype(dtype)
    if dt == torch.float16 and "gemma" in model_id.lower():
        print("[model] WARNING: fp16 with Gemma 3 can overflow; prefer bf16 or fp32", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(revision=revision, device_map="auto" if torch.cuda.is_available() else None, dtype=dt)
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
            if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    elif quant != "none":
        raise ValueError(f"quant must be none|4bit|8bit, got {quant!r}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, KeyError) as e:
        if "gemma-3" not in model_id.lower():
            raise
        from transformers import Gemma3ForConditionalGeneration
        print(f"[model] AutoModelForCausalLM failed ({str(e)[:80]!r}); using Gemma3ForConditionalGeneration", flush=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    info = {"dtype_resolved": str(dt).replace("torch.", ""), "quant": quant,
            "revision_resolved": resolve_revision(model_id, revision, model),
            "n_params_total": count_params(model_id, revision), "context_length": context_length(model.config)}
    return model, tok, info


def context_length(config):
    return (getattr(config, "max_position_embeddings", None)
            or getattr(getattr(config, "text_config", None), "max_position_embeddings", None))


def count_params(model_id, revision=None):
    """Total parameters (all towers), the number that counts against the 8B limit; quantisation does not change it.
    Counted from the safetensors headers of the downloaded checkpoint (exactly what is stored; tied matrices once).
    Fallback for .bin-only checkpoints (NLLB, which stores its tied embedding several times): unique tensors of the
    architecture on the meta device after tie_weights().
    Before 2026-10-24 this summed meta parameters WITHOUT tying and over-counted tied embeddings (Gemma-3-4B-it 4.97B
    instead of 4.30B, MADLAD-400-3B 3.46B instead of 2.94B, NLLB-600M 1.40B instead of 0.62B)."""
    try:
        import json
        import struct
        from pathlib import Path
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(model_id, revision=revision, allow_patterns=["*.safetensors"]))
        total = 0
        for f in folder.glob("*.safetensors"):
            with open(f, "rb") as fh:
                header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            for k, v in header.items():
                if k != "__metadata__":
                    n = 1
                    for d in v["shape"]:
                        n *= d
                    total += n
        if total:
            return total
        import transformers as T
        from accelerate import init_empty_weights
        cfg = T.AutoConfig.from_pretrained(model_id, revision=revision)
        for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"):
            try:
                with init_empty_weights():
                    m = getattr(T, cls).from_config(cfg)
                m.tie_weights()
                return sum({id(p): p.numel() for p in m.parameters()}.values())
            except (ValueError, KeyError):
                continue
    except Exception as e:  # informative only; never block a run on it
        print(f"[model] parameter count failed: {e}", flush=True)
    return None


def resolve_revision(model_id, revision, model):
    rev = getattr(model.config, "_commit_hash", None)
    if rev:
        return rev
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:
        return None


def render(tok, turns, chat, answer_word=None):
    """Exact text fed to the model. turns = ((role, text), ...) from prompts.build(); the last turn is the user turn
    being answered; assistant turns hold an example's answer label.
    answer_word set (label scoring): the reply is pre-filled so that the NEXT token is the label:
      chat: the model turn starts with "<answer_word>: " (examples' answers are written the same way)
      raw : the prompt already ends with "<answer_word>:"; a space is appended
    The trailing space matters: Gemma encodes " 2" as a space token + "2". Without the pre-fill Gemma-3-4B-it starts
    its reply with "වැඩිම ගැළපෙන පිළිතුර: ..." or "**" and only ~5% of the probability is on the labels; with it, ~100%.
    answer_word None (generation): no pre-fill."""
    if isinstance(turns, str):
        turns = (("user", turns),)
    if not chat:
        out = ""
        for role, text in turns:
            out += text if role == "user" else f" {text}\n\n"
        return out + (" " if answer_word else "")
    kw = dict(tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)
    if "enable_thinking" in (tok.chat_template or ""):
        kw["enable_thinking"] = False
    msgs = [{"role": r, "content": f"{answer_word}: {t}" if r == "assistant" and answer_word else t} for r, t in turns]
    return tok.apply_chat_template(msgs, **kw) + (f"{answer_word}: " if answer_word else "")


def label_token_ids(tok, labels):
    """Per label, the token id(s) of the bare label ('2'), which is what follows the pre-filled "...: ". Only if a
    tokenizer has no single-token bare label is ' 2' used instead."""
    out = []
    for d in labels:
        ids = {enc[0] for s in (d,) if len(enc := tok.encode(s, add_special_tokens=False)) == 1}
        if not ids:
            ids = {enc[0] for s in (" " + d,) if len(enc := tok.encode(s, add_special_tokens=False)) == 1}
        assert ids, f"label {d!r} has no single-token encoding"
        out.append(sorted(ids))
    return out


class LM:
    """A loaded model in one chat mode. Inputs are never truncated; non-finite logits raise.
      lm(prompts, labels, answer_word)     -> per prompt 4 log P(label | prompt + pre-fill)  (the scorer of scoring.py)
      lm.label_logprobs_text(texts, labels) -> the same on already rendered text (reasoning appended)
      lm.cloze(prompts, options, word)     -> per prompt and option (sum log P(option text | pre-fill), n tokens)
      lm.generate(prompts | texts, n)      -> greedy continuations
    """

    def __init__(self, model, tok, chat, batch_size=8, context=None, gen_batch_size=16):
        if chat and getattr(tok, "chat_template", None) is None:
            raise ValueError("chat=on but the tokenizer has no chat template")
        self.model, self.tok, self.chat = model, tok, chat
        self.batch_size, self.gen_batch_size = batch_size, gen_batch_size
        self.context = context or 10 ** 9
        self.max_len = 0
        self._label_ids = {}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def render(self, prompt, answer_word=None):
        return render(self.tok, prompt, self.chat, answer_word)

    def encode(self, texts):
        # chat templates already contain BOS; raw text gets it from the tokenizer
        ids = self.tok(list(texts), add_special_tokens=not self.chat)["input_ids"]
        longest = max(len(x) for x in ids)
        if longest > self.context:
            raise ValueError(f"input of {longest} tokens exceeds the context length {self.context}")
        self.max_len = max(self.max_len, longest)
        return ids

    def _forward(self, batch_ids, keep):
        """Left-padded forward pass; log-softmax of the last `keep` positions, shape (batch, keep, vocab)."""
        import torch
        enc = self.tok.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt").to(self.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        try:
            logits = self.model(**enc, position_ids=pos, logits_to_keep=keep).logits[:, -keep:].float()
        except TypeError:
            logits = self.model(**enc, position_ids=pos).logits[:, -keep:].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits (overflow?): use dtype bf16/fp32, not fp16")
        return torch.log_softmax(logits, dim=-1)

    def _batches(self, ids_all, size, desc):
        """Length-sorted batches (little padding), with a progress bar on stderr."""
        from tqdm.auto import tqdm
        order = sorted(range(len(ids_all)), key=lambda i: len(ids_all[i]))
        batches = [order[b:b + size] for b in range(0, len(order), size)]
        return tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True, mininterval=2,
                    disable=len(batches) < 2)

    def _oom_safe(self, fn, idx):
        """fn(idx) on one batch; on CUDA out-of-memory the batch is split in halves and retried (one question that
        does not fit still raises)."""
        import torch
        try:
            fn(idx)
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            print(f"[lm] out of memory on a batch of {len(idx)}: splitting", flush=True)
            half = len(idx) // 2
            self._oom_safe(fn, idx[:half])
            self._oom_safe(fn, idx[half:])

    def label_logprobs_text(self, texts, labels):
        import torch
        if labels not in self._label_ids:
            self._label_ids[labels] = label_token_ids(self.tok, labels)
        lab = self._label_ids[labels]
        ids_all = self.encode(texts)
        out = [None] * len(ids_all)

        def run(idx):
            lp = self._forward([ids_all[j] for j in idx], 1)[:, -1].cpu()
            for j, row in zip(idx, lp):
                out[j] = [torch.logsumexp(row[v], dim=0).item() for v in lab]

        with torch.no_grad():
            for idx in self._batches(ids_all, self.batch_size, f"score {len(ids_all)}"):
                self._oom_safe(run, idx)
        return out

    def __call__(self, prompts, labels, answer_word):
        return self.label_logprobs_text([self.render(p, answer_word) for p in prompts], labels)

    def cloze(self, prompts, options, answer_word):
        """options[i] = the 4 option texts of prompt i. Returns per prompt 4 (sum log-prob, token count)."""
        import torch
        ctx = self.encode([self.render(p, answer_word) for p in prompts])  # ends with "<answer_word>: "
        items = []  # (prompt index, option index, context ids, continuation ids)
        for i, opts in enumerate(options):
            for k, o in enumerate(opts):
                cont = self.tok.encode(o, add_special_tokens=False)
                items.append((i, k, ctx[i], cont))
        seqs = [c + o for _, _, c, o in items]
        self.max_len = max(self.max_len, max(len(s) for s in seqs))
        out = [[None] * 4 for _ in prompts]

        def run(idx):
            keep = max(len(items[j][3]) for j in idx) + 1
            lp = self._forward([seqs[j] for j in idx], keep)
            for row, j in enumerate(idx):
                i, k, _, cont = items[j]
                n = len(cont)
                # positions -n-1 .. -2 predict the n continuation tokens
                pred = lp[row, keep - n - 1:keep - 1]
                tgt = torch.tensor(cont, device=pred.device)
                out[i][k] = (pred.gather(1, tgt[:, None]).sum().item(), n)

        with torch.no_grad():
            for idx in self._batches(seqs, self.batch_size, f"cloze {len(prompts)}x4"):
                self._oom_safe(run, idx)
        return out

    def generate(self, prompts, max_new_tokens, rendered=False, sample_seed=None, temperature=0.7, top_p=0.95):
        """Greedy decoding (deterministic) by default. With sample_seed: nucleus sampling, reseeded per batch from
        sample_seed + the batch's first index, so a rerun on the same hardware gives the same texts.
        Returns the generated text only."""
        import torch
        texts = list(prompts) if rendered else [self.render(p) for p in prompts]
        ids_all = self.encode(texts)
        out = [None] * len(ids_all)
        pad = self.tok.pad_token_id
        if sample_seed is None:
            decoding = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            decoding = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=None)

        def run(idx):
            enc = self.tok.pad({"input_ids": [ids_all[j] for j in idx]}, padding=True, return_tensors="pt").to(self.device)
            if sample_seed is not None:
                torch.manual_seed(sample_seed + idx[0])
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=pad, **decoding)
            for row, j in enumerate(idx):
                out[j] = self.tok.decode(g[row, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        with torch.no_grad():
            for idx in self._batches(ids_all, self.gen_batch_size, f"generate {len(ids_all)} (<= {max_new_tokens} tokens)"):
                self._oom_safe(run, idx)
        return out


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def env_info():
    import platform
    info = {"python": platform.python_version(), "git_rev": git_rev()}
    for mod in ("torch", "transformers", "accelerate", "bitsandbytes", "peft"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def git_rev():
    """Commit hash, with '+dirty' when the working tree has uncommitted changes."""
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return None
