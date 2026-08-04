#!/usr/bin/env python3
"""Build one formal forward-transfer reference from multiple seed batches."""

from __future__ import annotations

import argparse
import json

from evaluation.single_task_baselines import aggregate_single_task_seed_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-directories", nargs="+", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--tail-size", type=int, default=5)
    args = parser.parse_args()

    result = aggregate_single_task_seed_batches(
        batch_directories=args.batch_directories,
        output_directory=args.output_directory,
        tail_size=args.tail_size,
    )
    print(
        json.dumps(
            {
                "output_directory": str(result.output_directory),
                "baseline_curves": str(result.curves_json_path),
                "summary": str(result.summary_json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
