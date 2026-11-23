# NLP-00 — Research protocol (DRAFT / NOT LOCKED)

Updated: 2026-11-23. **Status: DRAFT.** This is an executable decision record, not proof that task NLP-00 is complete.

## Research question

Does a classifier trained on the established Vietnamese hate-speech benchmark remain reliable on newer and linguistically altered Vietnamese social-media text, especially for the `HATE` class?

## Fixed study scope

- Primary official classification benchmark: ViHSD, classes `CLEAN`, `OFFENSIVE`, `HATE`.
- Primary comparison: TF-IDF + Linear SVM, PhoBERT, ViSoBERT and ViHateT5.
- Primary metric: Macro-F1. Also record per-class precision, recall and F1, `HATE` recall, confusion matrix, accuracy and support.
- Keep **benchmark reproduction**, **controlled robustness slices**, and **2025–2026 temporal holdout** separate. ViHOS span evaluation is not ViHSD document classification.
- Test/temporal data must not be used for training, early stopping, threshold/model/prompt choice or hyperparameter search.
- Error-analysis slices (e.g. body shaming, sarcasm, indirect abuse) are not new gold labels.

## Dataset / preprocessing decisions awaiting verification

| Item | State | Next evidence |
| --- | --- | --- |
| Official ViHSD dataset URL | Proposed: https://huggingface.co/datasets/uitnlp/vihsd | Read official card and dataset repo |
| Source revision/checksum | OPEN | Record exact revision/hash |
| License and access rights | OPEN | Confirm source terms with dataset auditor |
| Exact train/dev/test partitions | OPEN | Inspect official source files |
| Raw column schema / unique IDs | OPEN | Audit real dataset |
| Integer ID → class label mapping | OPEN | Verify from official source; never infer |
| Null, duplicate and split-leakage audit | OPEN | Emit audit report; do not silently resplit |
| PhoBERT word segmentation procedure | OPEN | Confirm official checkpoint/data preprocess |
| Primary random seed | PROPOSED: 42 | Group signoff before runs |
| PhoBERT max sequence length | OPEN | Train/dev length audit and truncation stats |

Store a machine-readable `data_manifest.json` beside local data (ignored by Git); use `docs/data_manifest.example.json` as template. Keep raw comments and personal information outside Git.

## Baselines

**SVM**: train-only TF-IDF word (1,2) + character `char_wb` (2,5), `LinearSVC`; consider `class_weight=None` versus `balanced` using dev only.

**PhoBERT**: validate official segmentation/tokenizer, checkpoint revision, label mapping and VRAM before full fine-tuning. Use dev Macro-F1 for checkpoint selection; record OOM/resource compromises.

## Reproducibility / reporting

Every run must retain commit + dirty status, dataset manifest, seed, preprocessing, library/CUDA versions, training config, command, runtime, logs, exit status, predictions and metrics. Literature scores must be marked as literature, not experiment outputs. Robustness transformations must retain provenance and whether they preserve meaning.

## Approval to lock NLP-00

Await group agreement on dataset audit, split, label IDs, seed, preprocessing, and any unresolved usage/license terms. Until then, this document remains **DRAFT**, and all model performance values remain unknown.
