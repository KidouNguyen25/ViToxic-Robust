"""Initial CLI: strict evaluation of a saved prediction CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from vitoxic_robust.data.schema import LABELS
from vitoxic_robust.evaluation.metrics import evaluate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vitoxic", description="ViToxic-Robust research CLI"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    evaluation = subcommands.add_parser(
        "evaluate", help="Validate one prediction CSV against frozen reference IDs"
    )
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--reference", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evaluate":
        try:
            metrics = evaluate(args.predictions, args.reference)
        except (ValueError, OSError, csv.Error) as exc:
            print(f"evaluation failed: {exc}", file=sys.stderr)
            return 2

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (args.output / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["label", "precision", "recall", "f1", "support"]
            )
            writer.writeheader()
            for label in LABELS:
                writer.writerow({"label": label, **metrics["per_class"][label]})

        print(
            f"{metrics['model_name']} on {metrics['split']} "
            f"(n={metrics['support']}): "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"HATE recall={metrics['hate_recall']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )
        print(f"Wrote {args.output / 'metrics.json'} and metrics.csv")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
