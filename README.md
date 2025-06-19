#  Fine‑Tuning BERT on SQuAD v2.0

This repo shows how I fine‑tuned **bert‑base‑uncased** for extractive question answering using the Hugging Face  ecosystem.

| Metric | Value* |
|-------|--------|
| Exact Match (dev) | 77.8 |
| F1 (dev) | 80.2 |
| Training time | 1 epoch ≈ 1 h (RTX 3060) |

\* Results after one epoch, batch = 8, lr = 3e‑5.

##  Repository layout
| Path | Purpose |
|------|---------|
| `train.py` | End‑to‑end training script (loads SQuAD v2, tokenises with sliding window, trains, evaluates, saves model) |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps large artefacts out of Git |
| `sample_predictions.md` | Five demo Q & A pairs produced by the model |

##  Quick start

```bash
git clone https://github.com/yourusername/bert-squad-finetuning.git
cd bert-squad-finetuning
python -m venv .env && source .env/bin/activate  # optional
pip install -r requirements.txt

# Train (single epoch default)
python train.py \
    --epochs 1 \
    --batch_size 8 \
    --model_name bert-base-uncased \
    --output_dir models/bert-squad
