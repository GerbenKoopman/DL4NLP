#!/usr/bin/env python3
"""
Analyze and visualize training and evaluation results.

- Reads training history CSVs in results/*/training_history_*.csv
- Reads evaluation summary CSVs in results/*/reptile_evaluation_summary_*.csv
- Reads baseline vs reptile comparison JSONs
- Produces plots under results/plots/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_files(patterns: List[str], root: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    # Deduplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_training_history(
    csv_path: Path,
    out_dir: Path,
    train_task_filter: Optional[list] = None,
    include_meta_average: bool = True,
) -> None:
    df = pd.read_csv(csv_path)
    title = f"Training History - {csv_path.stem.replace('training_history_', '')}"

    # Determine which columns to plot
    columns = [c for c in df.columns if c != "meta_step"]
    if not include_meta_average and "meta_average" in columns:
        columns.remove("meta_average")
    if train_task_filter:
        allowed = set(train_task_filter)
        columns = [c for c in columns if c in allowed]

    if not columns:
        logger.info(f"No training series to plot after filtering for {csv_path.name}")
        return

    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel("Meta Step")
    plt.ylabel("Score (BLEU/chrF blend)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_file = out_dir / f"{csv_path.stem}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.info(f"Saved {out_file}")


def plot_evaluation_summary(
    csv_path: Path,
    out_dir: Path,
    task_filter: Optional[None] = None,
    eval_type_filter: Optional[None] = None,
) -> None:
    df = pd.read_csv(csv_path)
    model = df["model"].iloc[0] if "model" in df.columns and not df.empty else "unknown"

    # Pivot by evaluation_type x task_type
    pivot = df.pivot_table(
        index="task_type", columns="evaluation_type", values="score", aggfunc="mean"
    )
    # Optional filtering
    if task_filter:
        keep_tasks = [t for t in pivot.index if t in set(task_filter)]
        pivot = pivot.loc[keep_tasks]
    if eval_type_filter:
        keep_cols = [c for c in pivot.columns if c in set(eval_type_filter)]
        pivot = pivot[keep_cols]
    pivot = pivot.sort_index()

    if pivot.empty:
        logger.info(f"No data to plot after filtering for {csv_path.name}")
        return

    plt.figure(figsize=(12, max(6, len(pivot) * 0.3)))
    pivot.plot(kind="barh", figsize=(12, max(6, len(pivot) * 0.3)))
    plt.xlabel("Score (BLEU/chrF blend)")
    plt.ylabel("Task Type")
    plt.title(f"Evaluation Summary - {model}")
    plt.tight_layout()

    out_file = out_dir / f"{csv_path.stem}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.info(f"Saved {out_file}")


def summarize_comparison(json_path: Path, out_dir: Path) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    comp = data.get("task_comparisons", {})
    rows = []
    for task, vals in comp.items():
        rows.append(
            {
                "task_type": task,
                "baseline_zero_shot": vals.get("baseline_zero_shot", 0.0),
                "reptile_zero_shot": vals.get("reptile_zero_shot", 0.0),
                "reptile_few_shot_5": vals.get("reptile_few_shot_5", 0.0),
                "zero_shot_improvement": vals.get("zero_shot_improvement", 0.0),
                "few_shot_improvement": vals.get("few_shot_improvement", 0.0),
            }
        )
    df = pd.DataFrame(rows).sort_values("few_shot_improvement", ascending=False)

    out_csv = out_dir / f"{json_path.stem}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}")

    # Plot improvements (side-by-side bars to avoid color blending)
    plt.figure(figsize=(10, max(5, len(df) * 0.3)))
    y = list(range(len(df)))
    bar_height = 0.38
    offset = bar_height / 2
    plt.barh(
        [i + offset for i in y],
        df["few_shot_improvement"],
        height=bar_height,
        color="#2a9d8f",
        label="Few-shot (5) Improvement",
    )
    plt.barh(
        [i - offset for i in y],
        df["zero_shot_improvement"],
        height=bar_height,
        color="#e76f51",
        label="Zero-shot Improvement",
    )
    plt.yticks(y, df["task_type"])
    plt.xlabel("Improvement over Baseline")
    plt.ylabel("Task Type")
    plt.title("Baseline vs Reptile Improvements")
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / f"{json_path.stem}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info(f"Saved {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and visualize DL4NLP results")
    parser.add_argument(
        "--results_dir", default="results", help="Path to results directory"
    )
    parser.add_argument(
        "--plots_dir", default="results/plots", help="Where to save plots"
    )
    # Training history filters
    parser.add_argument(
        "--train_tasks",
        default="all",
        help='Comma-separated training task series to include (e.g., be_en,en_be,meta_average) or "all"',
    )
    parser.add_argument(
        "--no_meta_average",
        action="store_true",
        help="Exclude meta_average from training history plots",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help='Comma-separated task_types to include (e.g., az_tr,be_uk) or "all"',
    )
    parser.add_argument(
        "--eval_types",
        default="all",
        help='Comma-separated evaluation types to include (e.g., zero_shot,few_shot_5,transfer_5) or "all"',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    ensure_dir(plots_dir)

    # 1) Training histories
    train_task_filter = (
        None
        if args.train_tasks.strip().lower() == "all"
        else [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    )
    include_meta_average = not args.no_meta_average
    train_csvs = find_files(["training_history_*.csv"], results_dir)
    if not train_csvs:
        logger.info("No training history CSVs found.")
    for csv_path in train_csvs:
        plot_training_history(
            csv_path,
            plots_dir,
            train_task_filter=train_task_filter,
            include_meta_average=include_meta_average,
        )

    # Build filters
    task_filter = (
        None
        if args.tasks.strip().lower() == "all"
        else [t.strip() for t in args.tasks.split(",") if t.strip()]
    )
    eval_type_filter = (
        None
        if args.eval_types.strip().lower() == "all"
        else [e.strip() for e in args.eval_types.split(",") if e.strip()]
    )

    # 2) Evaluation summaries
    eval_csvs = find_files(["reptile_evaluation_summary_*.csv"], results_dir)
    if not eval_csvs:
        logger.info("No evaluation summary CSVs found.")
    for csv_path in eval_csvs:
        plot_evaluation_summary(
            csv_path,
            plots_dir,
            task_filter=task_filter,
            eval_type_filter=eval_type_filter,
        )

    # 3) Baseline vs Reptile comparisons
    comp_jsons = find_files(["baseline_vs_reptile_*.json"], results_dir)
    if not comp_jsons:
        logger.info("No baseline vs reptile comparison JSONs found.")
    for json_path in comp_jsons:
        summarize_comparison(json_path, plots_dir)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
