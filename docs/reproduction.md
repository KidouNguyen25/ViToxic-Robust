# Reproduction guide — initialization stage

This guide only reproduces the **scaffold** and synthetic evaluator unit tests. No official ViHSD experiment has run.

## Windows PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m vitoxic_robust.cli --help
git diff --check
```

Dataset handling and training commands will be documented only after validated data-manifest and actual pipelines are implemented.

## Evaluator input contract

Reference CSV: `example_id,gold_label,split`, exactly one row per example.

Prediction CSV: `example_id,gold_label,predicted_label,model_name,split,seed`, exactly one row per example. All rows in one file should belong to the same named model, split and seed; scores must not be fabricated.

The `evaluate` command validates membership, labels and split/gold agreement and writes `metrics.json`, `metrics.csv` into the selected output directory. All outputs are local; place them under ignored `artifacts/`.

## Future experiment manifest

Store source revision, dataset splits, license review, git commit SHA and dirty status, runtime/environment/library versions, seed, model/checkpoint/tokenizer revisions, preprocessing version, command, duration, device/peak VRAM, logs, predictions and scores.
