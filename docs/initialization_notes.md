# Project initialization notes (technical reference)

Updated: 2026-11-23
Repository: `ViToxic-Robust`
Branch: `main`

> This document is technical setup and verification history, **not** a progress report authored of any other team member. Individual human work reports belong exclusively in `reports/`.

## Initial scaffold

- Python package using a `src/` layout and `pyproject.toml`.
- `README.md`, local `AGENTS.md`, research protocol draft, dataset manifest template, baseline configuration templates, reproduction notes and implementation plan.
- Strict CSV prediction evaluator: validates ID/label/split consistency, calculates Macro-F1, classwise metrics, HATE recall, accuracy and confusion matrix, and exports JSON/CSV metrics.
- Synthetic unit tests for evaluation edge cases. The synthetic scores are **not** actual ViHSD benchmark results.
- `.gitignore` excludes local data, artifacts, checkpoints, virtual environments and common secrets.

## Verified setup commands (2026-11-23)

| Command | Result |
| --- | --- |
| `python --version` | Python 3.12.10 |
| `python -m venv .venv` | Exit 0 |
| `.\\.venv\\Scripts\\python.exe -m pip install --no-deps -e .` | Exit 0; installed vitoxic-robust 0.1.0 |
| `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v` | 11 tests passed |
| `python -m compileall -q src tests` | Exit 0 |
| `.\\.venv\\Scripts\\python.exe -m vitoxic_robust.cli --help` | Exit 0 |
| `git diff --check` | Exit 0; does not check untracked files |
| `git check-ignore .venv/ artifacts/ data/` | All ignored |

## Known constraints

- Dataset access, official ViHSD schema/split, label ID mapping and usage permissions are not yet audited.
- The SVM and PhoBERT training pipelines are not implemented; no genuine model experiment has run.
- `NLP-00` protocol remains a draft, pending team agreement.
- `AGENTS.md` was initially excluded by local `.git/info/exclude`; it is deliberately included in the repository to version the contributor rules.
- At initialization, no Git commit was created automatically. Consult the repository history for the current committed state.

## Next engineering milestone

Audit ViHSD and lock the protocol, then implement and test the train-only SVM baseline followed by PhoBERT smoke checks.
