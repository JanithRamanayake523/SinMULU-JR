# Team SinBotics LoRA adapter (`adapter/`)

This is Team SinBotics' LoRA adapter for the **Sinhala MMLU Shared Task @ ICTer 2026**, on top of [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`). Code, inference script and predictions are in the main folder of this repo, [`SinhalaMMLU-SharedTask/team-sinbotics`](https://huggingface.co/SinhalaMMLU-SharedTask/team-sinbotics).

| | |
|---|---|
| Training data | Global-MMLU Sinhala (`CohereLabs/Global-MMLU`, `si`) + sinmulu_si (our Sri Lankan question collection): 25,799 training / 2,745 validation questions |
| Objective | the answer digit after a plain multiple-choice prompt with the reply pre-filled "Answer: " (loss on that token only) |
| LoRA | rank 16, alpha 32, dropout 0.05; attention, MLP and linear-attention projections of the 32 text layers; 30,474,240 parameters |
| Optimiser | AdamW, lr 1e-4, cosine, 3% warm-up, 32 questions per step, 1 epoch (807 steps), seed 42 |
| Hardware | 1× RTX 3090 24 GB, 8.8 h |
| Validation accuracy | 44.4% (base) → 54.8% |
| Parameters at inference | 4,205,751,296 (Qwen3.5-4B text model) + 30,474,240 (adapter), merged |

Files: `adapter_config.json` and `adapter_model.safetensors` (adapter), the tokenizer files of the base model, `train_config.json` and `train_log.jsonl` (training run).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = "Qwen/Qwen3.5-4B"; rev = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
tok = AutoTokenizer.from_pretrained(base, revision=rev)
model = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "SinhalaMMLU-SharedTask/team-sinbotics", subfolder="adapter").merge_and_unload()
```
The final system adds Qwen's thinking mode and a teacher-role English reasoning prompt; see `inference.py` in the main repo.
