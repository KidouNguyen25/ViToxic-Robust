# Implementation plan (2026-11-23)

This plan covers Toàn's tasks only. Coordination with Tùng (ViSoBERT), Trung (ViHateT5) and Quyền (robustness/temporal data) remains required.

| Stage | Dates | Task | Deliverable / acceptance |
| --- | --- | --- | --- |
| P0 | 23–24 Nov | NLP-00 | Versioned research protocol; dataset/split/label/source decisions explicitly locked or BLOCKED |
| P1a | 23–25 Nov | NLP-05 | Audit ViHSD; train-only TF-IDF + LinearSVC; saved prediction, config and metrics |
| P1b | 25–27 Nov | NLP-05 | PhoBERT segmentation and 4 GB GPU smoke run; fine-tuning if feasible, otherwise concrete blocker evidence |
| P2 | 28 Nov–4 Dec | NLP-10 | Integrate four adapters, strict same-example validation, shared metrics and comparison tables |
| P3 | 5–8 Dec | Integration | Controlled robustness and approved temporal tests, no test-set leakage |
| P4 | 9–11 Dec | NLP-13 | Reproducible report, slides, appendix, actual code/configs, risk and limitations |

## Current initialization scaffold (not yet committed)

- Repository instructions, packaging, documentation, safe ignore rules and YAML baseline templates.
- Minimal standard-library strict evaluator CLI + synthetic tests, **not** real benchmark scores.
- No downloads, model weights, test-set manipulation, network scraping, or claims of model readiness.

## High-priority blockers

1. Confirm actual ViHSD schema, label IDs, splits and licence/terms before training.
2. Group approval for protocol and seed.
3. PhoBERT word segmentation and available VRAM, test via a tiny smoke job.
4. Teammate output compatibility; no assumption their models have already run.

## Completion definition

A task is only Done when source files, tested commands, logs, metrics, evidence paths and reproduction instructions exist. Notebook screenshots or paper-reported scores alone do not qualify.
