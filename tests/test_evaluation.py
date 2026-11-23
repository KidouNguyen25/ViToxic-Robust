"""Synthetic tests: no real ViHSD comments, labels or scores are involved."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vitoxic_robust.cli import main
from vitoxic_robust.data.schema import LABELS
from vitoxic_robust.evaluation.metrics import evaluate


def save_csv(path: Path, columns: list[str], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root / "reference.csv"
        self.predictions = self.root / "predictions.csv"
        self.reference_rows = [
            ("a", "CLEAN", "test"),
            ("b", "OFFENSIVE", "test"),
            ("c", "HATE", "test"),
        ]
        self.pred_rows = [
            ("a", "CLEAN", "CLEAN", "smoke", "test", "42"),
            ("b", "OFFENSIVE", "HATE", "smoke", "test", "42"),
            ("c", "HATE", "HATE", "smoke", "test", "42"),
        ]
        self.write_both()

    def write_both(self) -> None:
        save_csv(self.reference, ["example_id", "gold_label", "split"], self.reference_rows)
        save_csv(
            self.predictions,
            ["example_id", "gold_label", "predicted_label", "model_name", "split", "seed"],
            self.pred_rows,
        )

    def test_label_schema_is_three_unique_classes(self) -> None:
        self.assertEqual(LABELS, ("CLEAN", "OFFENSIVE", "HATE"))
        self.assertEqual(len(set(LABELS)), 3)

    def test_hand_checked_macro_f1_hate_recall_and_confusion(self) -> None:
        scores = evaluate(self.predictions, self.reference)
        self.assertAlmostEqual(scores["macro_f1"], 5 / 9)
        self.assertEqual(scores["hate_recall"], 1.0)
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)
        self.assertEqual(scores["confusion_matrix"], [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(scores["per_class"]["OFFENSIVE"]["support"], 1)

    def test_missing_example_rejected(self) -> None:
        self.pred_rows.pop()
        self.write_both()
        with self.assertRaisesRegex(ValueError, "example_id mismatch"):
            evaluate(self.predictions, self.reference)

    def test_extra_example_rejected(self) -> None:
        self.pred_rows.append(("d", "CLEAN", "CLEAN", "smoke", "test", "42"))
        self.write_both()
        with self.assertRaisesRegex(ValueError, "example_id mismatch"):
            evaluate(self.predictions, self.reference)

    def test_duplicate_prediction_id_rejected(self) -> None:
        self.pred_rows.append(self.pred_rows[0])
        self.write_both()
        with self.assertRaisesRegex(ValueError, "duplicate example_id"):
            evaluate(self.predictions, self.reference)

    def test_unknown_label_rejected(self) -> None:
        self.pred_rows[0] = ("a", "CLEAN", "MAYBE", "smoke", "test", "42")
        self.write_both()
        with self.assertRaisesRegex(ValueError, "invalid predicted_label"):
            evaluate(self.predictions, self.reference)

    def test_wrong_gold_rejected(self) -> None:
        self.pred_rows[0] = ("a", "HATE", "CLEAN", "smoke", "test", "42")
        self.write_both()
        with self.assertRaisesRegex(ValueError, "gold_label differs"):
            evaluate(self.predictions, self.reference)

    def test_wrong_split_rejected(self) -> None:
        self.pred_rows[0] = ("a", "CLEAN", "CLEAN", "smoke", "dev", "42")
        self.write_both()
        with self.assertRaisesRegex(ValueError, "split differs"):
            evaluate(self.predictions, self.reference)

    def test_mixed_models_rejected(self) -> None:
        self.pred_rows[1] = ("b", "OFFENSIVE", "HATE", "other", "test", "42")
        self.write_both()
        with self.assertRaisesRegex(ValueError, "mix multiple values of model_name"):
            evaluate(self.predictions, self.reference)

    def test_cli_writes_structured_metrics(self) -> None:
        destination = self.root / "results"
        result = main([
            "evaluate", "--predictions", str(self.predictions),
            "--reference", str(self.reference), "--output", str(destination),
        ])
        self.assertEqual(result, 0)
        scores = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(scores["macro_f1"], 5 / 9)
        self.assertTrue((destination / "metrics.csv").is_file())

    def test_cli_bad_predictions_do_not_create_outputs(self) -> None:
        self.pred_rows[0] = ("a", "CLEAN", "UNKNOWN", "smoke", "test", "42")
        self.write_both()
        destination = self.root / "invalid_results"
        self.assertEqual(main([
            "evaluate", "--predictions", str(self.predictions),
            "--reference", str(self.reference), "--output", str(destination),
        ]), 2)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
