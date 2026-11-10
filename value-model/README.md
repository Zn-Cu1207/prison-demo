# Model training and evaluation

This directory contains the value-model and dialogue-model pipeline used by the NPC simulation. Model weights, generated training data, teacher credentials, and training outputs are excluded from this repository.

## Value-model pipeline

| Script | Role |
| --- | --- |
| `gen_situations.py` | Generate synthetic scenarios from the simulation's context and action definitions |
| `annotate_rank.py` / `annotate.py` | Obtain preferences from a configured teacher model |
| `train_value.py` | Train Qwen2.5-1.5B with LoRA, a scalar head, and confidence-weighted Bradley–Terry loss |
| `merge_lora.py` | Merge the trained adapter for inference |
| `eval.py` | Evaluate held-out pairs and behavioral probes |
| `value_server.py` | Serve action scores over HTTP |
| `test_consistency.py` | Check shared context/input construction |

Install the backend dependencies plus `requirements-train.txt` in a suitable training environment. Hardware, CUDA, PyTorch, and quantization support must match your installation.

Generate scenarios without calling an external model:

```bash
cd value-model
python gen_situations.py --n 1500 --out data/situations.jsonl
```

For teacher annotation, copy `.env.example` to `.env` and replace the endpoint, model name, and API key with values for your provider. The example endpoint is deliberately a placeholder. Annotation calls can incur provider charges.

```bash
python annotate_rank.py --in data/situations.jsonl --out data/preferences.jsonl --model YOUR_TEACHER_MODEL --limit 5
```

Inspect the small annotation run before increasing its size. Once valid preference data is available:

```bash
python train_value.py --pref data/preferences.jsonl --base Qwen/Qwen2.5-1.5B-Instruct --out runs/adapter --epochs 3
python merge_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/adapter --out merged/value-1p5b
python eval.py --model merged/value-1p5b --pref data/preferences.jsonl
python value_server.py --model merged/value-1p5b --port 8009
```

Use identical preference files, validation fraction, and seed for training and pairwise evaluation. The split groups pairs by `situation_id`.

## Dialogue adaptation

`gen_convo_sft.py`, `train_sft.py`, `merge_sft.py`, and `eval_convo.py` cover dialogue-data preparation, QLoRA/SFT, adapter merging, and evaluation. Use each script's `--help` for its inputs. The original fine-tuned dialogue checkpoint and its data are not distributed here.

The backend defaults to the local Ollama model tag `prison-convo:q4`; this is a project-specific tag, not a model that a fresh Ollama installation can automatically download. Supply the corresponding model yourself, or set `LOCAL_LLM_MODEL` to a model you already have. A base model fallback does not reproduce the fine-tuned dialogue behavior.
