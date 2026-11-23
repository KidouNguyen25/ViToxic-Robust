"""Strict, dependency-free document-classification evaluation.

Rows are keyed by stable example_id; never silently drop unmatched examples.
"""
from __future__ import annotations

import csv
from pathlib import Path

from vitoxic_robust.data.schema import LABELS


REFERENCE_COLUMNS = ("example_id", "gold_label", "split")
PREDICTION_COLUMNS = (
    "example_id", "gold_label", "predicted_label", "model_name", "split", "seed"
)


def _load_csv(path: str | Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"{path}: empty CSV or missing header")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path}: duplicate column names")
        missing = set(required) - set(columns)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        rows = []
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"{path}:{line_number}: too many CSV fields")
            row = {key: (value or "").strip() for key, value in raw.items()}
            if any(not row[field] for field in required):
                raise ValueError(f"{path}:{line_number}: empty required field")
            example_id = row["example_id"]
            if example_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate example_id {example_id!r}")
            seen_ids.add(example_id)
            for label_key in ("gold_label", "predicted_label"):
                if label_key in required and row[label_key] not in LABELS:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {label_key} {row[label_key]!r}"
                    )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no examples")
    return rows


def evaluate(predictions_path: str | Path, reference_path: str | Path) -> dict:
    """Evaluate exactly one model/seed/split against a frozen reference CSV."""
    reference = _load_csv(reference_path, REFERENCE_COLUMNS)
    predictions = _load_csv(predictions_path, PREDICTION_COLUMNS)
    ref_by_id = {row["example_id"]: row for row in reference}
    pred_by_id = {row["example_id"]: row for row in predictions}

    missing = set(ref_by_id) - set(pred_by_id)
    extra = set(pred_by_id) - set(ref_by_id)
    if missing or extra:
        raise ValueError(
            f"prediction/reference example_id mismatch: "
            f"{len(missing)} missing, {len(extra)} unexpected"
        )

    for example_id, pred in pred_by_id.items():
        gold = ref_by_id[example_id]
        for column in ("gold_label", "split"):
            if pred[column] != gold[column]:
                raise ValueError(f"{example_id}: {column} differs from reference")

    # Cross-split or cross-model aggregates must use a separate experiment adapter.
    for column in ("model_name", "split", "seed"):
        if len({row[column] for row in predictions}) != 1:
            raise ValueError(f"predictions mix multiple values of {column}")
    if len({row["split"] for row in reference}) != 1:
        raise ValueError("reference mixes multiple split values")

    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for row in predictions:
        gold = ref_by_id[row["example_id"]]["gold_label"]
        matrix[index[gold]][index[row["predicted_label"]]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(matrix[j][i] for j in range(len(LABELS)))
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(predictions)
    return {
        "model_name": predictions[0]["model_name"],
        "split": predictions[0]["split"],
        "seed": predictions[0]["seed"],
        "support": total,
        "labels": list(LABELS),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
        "hate_recall": per_class["HATE"]["recall"],
        "accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / total,
        "confusion_matrix": matrix,  # rows=gold, columns=predicted in labels order
        "per_class": per_class,
    }
