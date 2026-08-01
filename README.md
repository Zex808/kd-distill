# kd-distill

> **API/Ollama Teacher → Local GPU Student** via Hinton Knowledge Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1 · Harvest                                           │
│  prompts.jsonl ──► API (OpenAI / Anthropic / Ollama) ──►   │
│                    teacher_data.jsonl                        │
│                    {prompt, response, logprobs?}            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  Step 2 · Train (local GPU)                                 │
│                                                             │
│  L = α · T² · KL(student_soft ‖ teacher_soft)              │
│    + (1–α) · CE(student, hard_labels)                       │
│                                                             │
│  teacher_soft: sparse top-k logprobs (OpenAI)               │
│             or response CE only (Anthropic/Ollama)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  Step 3 · Evaluate                                          │
│  ROUGE-L / BERTScore student vs teacher                     │
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# Linux/Mac
chmod +x setup.sh && ./setup.sh

# Windows
.\setup.ps1
```

Opens Gradio UI at **http://localhost:7860**

---

## Ollama (free, no API key)

```bash
# Install: https://ollama.com
ollama pull qwen2.5:3b   # ~2GB — recommended teacher
ollama pull llama3.2:3b  # alt

# Verify
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:3b","prompt":"test","stream":false}' | jq .response
```

Then in UI: Provider=`ollama`, Model=`qwen2.5:3b`.

## Soft-label KD notes

| Provider | Logprobs | KD Mode |
|---|---|---|
| OpenAI | ✅ top-k via `logprobs` param | Sparse soft-label KD |
| Anthropic | ❌ not exposed | Response-only (CE) |
| Ollama | ❌ not exposed | Response-only (CE) |

**Cross-tokenizer caveat:** OpenAI uses `tiktoken` BPE; student models use their own tokenizers. `kd_trainer.py` does best-effort string matching to align vocabularies. Works well for `gpt2`-family students; degrades for DistilBERT (WordPiece). See `kd_trainer.py:build_sparse_teacher_targets()` for details.

## Config reference

| Key | Default | Notes |
|---|---|---|
| `teacher.temperature` | `1.0` | Higher = softer target distribution |
| `training.alpha` | `0.7` | KD loss weight; `1-α` = CE weight |
| `training.temperature` | `4.0` | Distillation temperature (Hinton 2015) |
| `training.grad_accum` | `4` | Effective batch = `batch_size × grad_accum` |
| `training.fp16` | `true` | Disable if GPU doesn't support |

## W&B tracking

```bash
pip install wandb
wandb login
```

Set `wandb.enabled: true` in `config.yaml`. Dashboard at https://wandb.ai

## CLI (no UI)

```bash
# Harvest only
python harvest.py

# Train only
python kd_trainer.py

# Eval only
python eval.py --checkpoint ./checkpoints/final --data data/teacher_data.jsonl
```

## Project structure

```
kd-distill/
├── app.py              Gradio UI (main entry point)
├── harvest.py          Teacher API → JSONL dataset
├── kd_trainer.py       Hinton KD loss + HF student training
├── eval.py             ROUGE-L / BERTScore scoring
├── config.yaml         All hyperparameters
├── examples/
│   └── prompts.jsonl   Sample prompts to get started
└── .github/
    └── workflows/
        └── lint.yml    Ruff lint CI
```

## License

MIT — see [LICENSE](LICENSE)
