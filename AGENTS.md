# ViToxic-Robust — contributor and agent instructions

This is an INT3406 research repository for Vietnamese toxic speech detection. Read `docs/research_protocol.md` and `docs/implementation_plan.md` before modifying experiments.

## Documentation language
- Write repository documentation in **English** (including `README.md`, `docs/`, and any future human-authored reports in `reports/`). Preserve existing technical names, commands, label codes, and identifiers.
- Do not translate quoted source data or research examples; keep their original text where required.

## Scope and ownership
- Use the official ViHSD three-class task: `CLEAN`, `OFFENSIVE`, `HATE`. Do not add ad hoc gold labels.
- Toàn owns protocol (`NLP-00`), SVM + PhoBERT baselines (`NLP-05`), shared evaluation (`NLP-10`), and final report/slides (`NLP-13`).
- ViSoBERT is Tùng's work, ViHateT5 Trung's, and robustness/temporal data Quyền's. Integrate via documented adapters; do not silently overwrite teammate work.
- No UI, deployment, database, API backend, or newly scraped dataset is in scope.

## Scientific integrity
- Inspect the official ViHSD dataset card and source files before locking column names, integer label IDs, split sizes, revisions or license. Unknowns must remain explicitly UNKNOWN.
- Preserve official train/dev/test partitions. Never train, tune, choose thresholds/checkpoints or select prompts on test or temporal samples.
- Keep raw text, stable example IDs, preprocessing versions and train-only feature fitting. Do not normalize away linguistic variants in primary robustness analyses.
- Separate official benchmark scores, robustness scores, temporal scores and literature-reported scores.
- Macro-F1 is primary; also report classwise P/R/F1, HATE recall, support, confusion matrix, accuracy, seeds and reproducibility metadata.
- Predictions must preserve `example_id, gold_label, predicted_label, model_name, split, seed` and match the locked reference IDs exactly.
- Every claimed result requires config, dataset revision, commit SHA, versions, logs, predictions and metrics. Never invent experimental results.

## Engineering and security
- Favor scripts/modules over notebook-only logic, small changes with tests, and deterministic commands.
- Keep raw comments, private identifiers, tokens, .env files and checkpoints outside Git; review `.gitignore`.
- Run `python -m unittest discover -s tests -v` and `git diff --check` for initial scaffold changes; later implement and document model-specific checks.
- Git messages use `type(scope): detail` with English detail and include the NLP task ID in the commit/PR description.
- Do not label a task completed until a saved artifact and evidence exist.

## Reports and activity records
- `reports/` is reserved for dated work/progress reports authored by Toàn or other identifiable human team members, describing work those individuals actually performed. Do not create an assistant/agent self-report or attribute agent work to a human.
- Only write or edit a person's report when explicitly requested and supported by that person's actual work evidence. Do not invent activities, results, authorship or task completion.
- Keep automated initialization notes, tool execution history and technical verification records in `docs/` (for example `docs/initialization_notes.md`), or summarize them directly in chat; never place them in `reports/`.

The initial scaffold has no downloaded dataset, trained model, or real benchmark result. Protocol decisions marked OPEN must be resolved with the team.
