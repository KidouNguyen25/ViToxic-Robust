# ViToxic-Robust

**INT3406 — Vietnamese Toxic Speech Detection Robustness**

Study whether classifiers trained on the official Vietnamese Hate Speech Detection benchmark (ViHSD) retain their performance under spelling errors, missing diacritics, teencode, repeated characters, code-switching, filter evasion and indirect attacks.

**Status (2026-11-23):** initialized research scaffold. **No dataset has been downloaded or experiment executed on ViHSD.** Do not interpret synthetic smoke-test results as benchmark evidence.

## Research scope

- **Gold labels:** `CLEAN`, `OFFENSIVE`, `HATE` (three-class document classification).
- **Official benchmark:** [ViHSD](https://huggingface.co/datasets/uitnlp/vihsd), subject to checking dataset access, label IDs, licensing and official splits.
- **Baselines planned:** word+character TF-IDF / Linear SVM and PhoBERT.
- **Comparators planned:** ViSoBERT (Tùng), ViHateT5 (Trung), through the shared evaluator.
- **Primary metric:** Macro-F1; report `HATE` recall, per-class P/R/F1, confusion matrix and accuracy.
- **Three separate evaluations:** official ViHSD benchmark, robustness slices, and independently approved 2025–2026 temporal data. Temporal samples are test-only.

See [protocol draft](docs/research_protocol.md), [implementation plan](docs/implementation_plan.md), [reproduction guide](docs/reproduction.md) and [technical initialization notes](docs/initialization_notes.md).

**Documentation language:** Write repository documentation in English. Keep technical identifiers, commands, and original research data unchanged.

## Setup

Prerequisite: Python **3.10+**. From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m vitoxic_robust.cli --help
```

The base install has **no heavy ML libraries**. Future optional baseline dependencies can be added when data and training requirements are validated.

## Working CLI (scaffold)

```powershell
python -m vitoxic_robust.cli evaluate --predictions path\to\predictions.csv --reference path\to\reference.csv --output artifacts\runs\local_check
```

Reference CSV: `example_id,gold_label,split`. Predictions CSV: `example_id,gold_label,predicted_label,model_name,split,seed`. The evaluator requires **the same unique IDs, gold labels and split membership** in both inputs; it rejects incomplete or inconsistent files and writes `metrics.json` and `metrics.csv`. It does not yet implement cross-model aggregation or robustness slices.

## Repository layout

- `configs/`: **planned** SVM/PhoBERT/evaluation YAML templates, not working training commands yet.
- `src/vitoxic_robust/`: source package; initially strict evaluation and CLI only.
- `tests/`: synthetic unit/regression tests. No real benchmark data.
- `docs/`: research protocol, data-manifest template, plan, reproduction instructions and technical initialization notes.
- `reports/`: reserved for progress/work reports from Toàn and other human team members about the work they actually performed. Do not store AI/agent activity reports here.
- `artifacts/`, `data/`: local-only outputs and raw datasets; ignored by Git.

## Milestones

`NLP-00` protocol → `NLP-05` SVM/PhoBERT baselines → `NLP-10` four-model evaluation → `NLP-13` final report/slides. Task status on Notion has not been independently verified at scaffold time.

**Scientific rule:** do not use the benchmark test split or frozen temporal sample for hyperparameter selection, data cleaning decisions, checkpoint selection, or model comparison-driven tuning.
