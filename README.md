# ViToxic-Robust

Vietnamese toxic-comment classification research. This repository contains a reproducible **binary toxicity baseline**, not a completed robust/implicit-attack detector.

## Data and model

- Dataset: [UIT-ViCTSD](https://huggingface.co/datasets/tarudesu/ViCTSD), 10,000 Vietnamese comments (train 7,000 / validation 2,000 / test 1,000).
- Input column: `Comment`. Classification target: `Toxicity` (0 = non-toxic, 1 = toxic). The separate `Constructiveness` field is not used.
- Base model: [Multilingual DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased).
- Validation and held-out test sets remain separate. Accuracy, macro-F1 and per-class test metrics are exported.
- The ViCTSD authors limit the provided dataset to research purposes; review the original dataset terms before redistributing or commercial use.

Dataset bytes are **not committed to GitHub**. The training script fetches the current dataset revision directly from Hugging Face each run.

## Run on GitHub Actions (CPU smoke test)

1. Merge the training setup PR into the default branch.
2. Open **Actions → NLP training from Hugging Face → Run workflow**.
3. Leave the defaults of 256 training / 64 validation / 64 test samples and one epoch. This only tests end-to-end execution, **not model quality**.
4. Inspect the job logs and download `vitoxic-nlp-model` under **Artifacts**. It contains `outputs/model/` and `outputs/metrics.json`.

The standard GitHub-hosted runner is CPU-only. For full training, use the script on a GPU machine. Setting sample limits to 0 uses the full dataset, but doing that on GitHub's standard CPU runner may be very slow or exceed the 90-minute job limit.

## Run locally with GPU (Windows/Linux)

Use Python 3.11 or 3.12. Install an appropriate CUDA-enabled PyTorch build from [PyTorch's installation guide](https://pytorch.org/get-started/locally/) when using an NVIDIA GPU.

```bash
python -m pip install -r requirements.txt
python scripts/train.py
```

Override settings via environment variables (PowerShell example):

```powershell
$env:MAX_TRAIN_SAMPLES="512"
$env:MAX_EVAL_SAMPLES="128"
$env:MAX_TEST_SAMPLES="128"
$env:EPOCHS="1"
python scripts/train.py
```

For full training, use `MAX_TRAIN_SAMPLES=0`, `MAX_EVAL_SAMPLES=0`, `MAX_TEST_SAMPLES=0`. Output is written to `outputs/`, which is ignored by Git.

## Hugging Face authentication

The public ViCTSD dataset is readable without a token. For private/gated datasets, or to publish model weights, create a Hugging Face token and store it in **GitHub → Settings → Secrets and variables → Actions → New repository secret**, named `HF_TOKEN`. A gated dataset also requires accepting the dataset's access conditions with that Hugging Face account.

GitHub **Actions variables** (non-secret) supported:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_DATASET_ID` | `tarudesu/ViCTSD` | Dataset repository |
| `HF_DATASET_REVISION` | `main` | Commit SHA/tag/branch for dataset provenance |
| `MODEL_NAME` | `distilbert/distilbert-base-multilingual-cased` | Pretrained model |
| `HF_MODEL_REPO_ID` | unset | If set, upload the resulting model to a **private** Hugging Face model repo (requires `HF_TOKEN` with write access) |

**Important:** Dataset changes on Hugging Face do not trigger GitHub Actions automatically. Each manual workflow run downloads the requested dataset revision. Set `HF_DATASET_REVISION` to a commit SHA for reproducible experiments; `main` may change over time.

The preprocessing in `scripts/train.py` expects the ViCTSD schema and 2 classes. When switching to ViHSD, a three-class dataset or other NLP task, update the input fields, label mapping, number of labels and split names first.

## Research next steps

This is only a functional baseline. ViToxic-Robust still needs at least: Vietnamese-domain baselines (PhoBERT/ViSoBERT), normalization for slang and obfuscated language, targeted adversarial evaluation, class imbalance analysis, duplicate/leakage audits, experiment tracking and a fixed held-out benchmark.
