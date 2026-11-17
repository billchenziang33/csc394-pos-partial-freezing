"""Run a paired, repeated-seed comparison and summarize its variability."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-dev", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--runs-dir", type=Path, default=Path("seed_runs"))
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        result_path = args.runs_dir / f"seed_{seed}.json"
        command = [
            sys.executable,
            "run_experiment.py",
            "--max-train", str(args.max_train),
            "--max-dev", str(args.max_dev),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(seed),
            "--results", str(result_path),
        ]
        print(f"Running paired experiment for seed {seed}...", flush=True)
        subprocess.run(command, check=True)
        runs.append(json.loads(result_path.read_text(encoding="utf-8")))

    summary = {}
    for condition in ("full_finetuning", "partial_freezing"):
        rows = [
            next(row for row in run["results"] if row["condition"] == condition)
            for run in runs
        ]
        scores = [row["dev_word_accuracy"] for row in rows]
        summary[condition] = {
            "mean_dev_word_accuracy": statistics.mean(scores),
            "sample_stdev_dev_word_accuracy": (
                statistics.stdev(scores) if len(scores) > 1 else None
            ),
            "mean_elapsed_seconds": statistics.mean(
                row["elapsed_seconds"] for row in rows
            ),
            "trainable_parameters": rows[0]["trainable_parameters"],
        }

    payload = {
        "model": runs[0]["model"],
        "dataset": runs[0]["dataset"],
        "seeds": args.seeds,
        "max_train_sentences": args.max_train,
        "max_dev_sentences": args.max_dev,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "frozen_components": runs[0]["frozen_components"],
        "device": runs[0]["device"],
        "runs": runs,
        "summary": summary,
        "summary_note": "Mean and sample standard deviation describe these paired seeds and seed-specific data samples; they do not establish generalization to other languages or datasets.",
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved aggregated results to {args.results}")


if __name__ == "__main__":
    main()
