"""Aggregate aligned learning curves from a CW10 single-task batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from crl_cw.evaluation.single_task_baselines import (
    aggregate_single_task_baselines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ten independent CW10 single-task SAC baselines."
        )
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help=(
            "Directory created by "
            "scripts/train_cw10_single_task_baselines.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional aggregate output directory.",
    )
    parser.add_argument(
        "--tail-size",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    if args.tail_size <= 0:
        parser.error("--tail-size must be positive.")
    return args


def main() -> None:
    args = parse_args()
    result = aggregate_single_task_baselines(
        batch_directory=Path(args.batch_dir),
        output_directory=(
            None if args.output_dir is None else Path(args.output_dir)
        ),
        tail_size=args.tail_size,
    )

    print("=" * 76)
    print("CW10 single-task baselines aggregated")
    print("=" * 76)
    print(f"Output directory         : {result.output_directory}")
    print(f"Baseline curves JSON     : {result.curves_json_path}")
    print(f"Long curves CSV          : {result.long_curves_csv_path}")
    print(f"Stochastic success CSV   : {result.stochastic_success_csv_path}")
    print(f"Task summaries CSV       : {result.task_summaries_csv_path}")
    print(f"Aggregate summary JSON   : {result.summary_json_path}")


if __name__ == "__main__":
    main()
