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

from paths import paths

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
    task_filter: Optional[List[str]] = None,
    eval_type_filter: Optional[List[str]] = None,
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
    plt.yticks(y, df["task_type"].tolist())
    plt.xlabel("Improvement over Baseline")
    plt.ylabel("Task Type")
    plt.title("Baseline vs Reptile Improvements")
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / f"{json_path.stem}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info(f"Saved {out_png}")


def plot_ablation_comparison(
    ablation_csv: Path, out_dir: Path, metric: str = "transfer_5"
) -> None:
    """Plot ablation study comparison across configurations
    
    Args:
        ablation_csv: Path to aggregated_results_*.csv from ablation study
        out_dir: Output directory for plots
        metric: Evaluation type to focus on (e.g., "transfer_5", "few_shot_5")
    """
    df = pd.read_csv(ablation_csv)
    
    # Filter to target metric
    df_metric = df[df["evaluation_type"] == metric].copy()
    
    if df_metric.empty:
        logger.warning(f"No data for metric {metric} in {ablation_csv.name}")
        return
    
    # Aggregate by config (mean across tasks)
    config_scores = (
        df_metric.groupby("config_name")["score"].mean().sort_values(ascending=False)
    )
    
    # Plot 1: Overall comparison
    plt.figure(figsize=(14, max(6, len(config_scores) * 0.3)))
    colors = plt.cm.viridis(range(len(config_scores)))
    plt.barh(range(len(config_scores)), config_scores.values, color=colors)
    plt.yticks(range(len(config_scores)), config_scores.index, fontsize=8)
    plt.xlabel(f"Mean Score ({metric})")
    plt.ylabel("Configuration")
    plt.title(f"Ablation Study: {metric.replace('_', ' ').title()} Performance")
    plt.tight_layout()
    
    out_file = out_dir / f"ablation_comparison_{metric}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.info(f"Saved {out_file}")
    
    # Plot 2: Heatmap by task type
    pivot = df_metric.pivot_table(
        index="config_name", columns="task_type", values="score", aggfunc="mean"
    )
    
    if not pivot.empty:
        plt.figure(figsize=(10, max(8, len(pivot) * 0.4)))
        vmin = float(pivot.values.min())
        vmax = float(pivot.values.max())
        # Avoid zero range for constant matrices
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Score")
        plt.yticks(range(len(pivot)), pivot.index, fontsize=7)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.title(f"Ablation Study Heatmap: {metric.replace('_', ' ').title()}")
        plt.tight_layout()
        
        out_file = out_dir / f"ablation_heatmap_{metric}.png"
        plt.savefig(out_file, dpi=200)
        plt.close()
        logger.info(f"Saved {out_file}")


def plot_ablation_factors(ablation_csv: Path, out_dir: Path) -> None:
    """Plot effect of individual ablation factors
    
    Analyzes meta_lr, inner_steps, support_size, adapter_mode effects
    """
    df = pd.read_csv(ablation_csv)
    
    # Parse config names to extract factors (assumes naming: metaLR0.10_inner3_support5_az_en_bleu0.60_seed42)
    def parse_config(config_name: str) -> dict:
        parts = config_name.split("_")
        factors = {}
        for part in parts:
            if part.startswith("metaLR"):
                factors["meta_lr"] = float(part.replace("metaLR", ""))
            elif part.startswith("inner"):
                factors["inner_steps"] = int(part.replace("inner", ""))
            elif part.startswith("support"):
                factors["support_size"] = int(part.replace("support", ""))
            elif part in ["az_en", "be_en", "all"]:
                factors["adapter_mode"] = part
            elif part.startswith("bleu"):
                factors["bleu_weight"] = float(part.replace("bleu", ""))
        return factors
    
    # Add factor columns
    factor_data = df["config_name"].apply(parse_config).apply(pd.Series)
    df = pd.concat([df, factor_data], axis=1)
    
    # Focus on transfer_5 for ablation analysis
    df_transfer = df[df["evaluation_type"] == "transfer_5"].copy()
    
    if df_transfer.empty:
        logger.warning(f"No transfer_5 data in {ablation_csv.name}")
        return
    
    # Plot effects of each factor
    factors_to_plot = ["meta_lr", "inner_steps", "support_size", "adapter_mode"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, factor in enumerate(factors_to_plot):
        if factor not in df_transfer.columns:
            continue
        
        # Aggregate scores by factor value
        factor_scores = (
            df_transfer.groupby(factor)["score"].agg(["mean", "std"]).reset_index()
        )
        
        ax = axes[idx]
        x_vals = range(len(factor_scores))
        ax.bar(x_vals, factor_scores["mean"], yerr=factor_scores["std"].fillna(0.0), capsize=5)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(factor_scores[factor], rotation=0 if idx < 2 else 45)
        ax.set_xlabel(factor.replace("_", " ").title())
        ax.set_ylabel("Mean Transfer Score")
        ax.set_title(f"Effect of {factor.replace('_', ' ').title()}")
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_file = out_dir / "ablation_factor_effects.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.info(f"Saved {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and visualize DL4NLP results")
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
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Generate ablation study plots from results/ablation/",
    )
    parser.add_argument(
        "--ablation_metric",
        default="transfer_5",
        help="Metric to focus on for ablation comparison (default: transfer_5)",
    )
    args = parser.parse_args()

    results_dir = paths.results_dir
    plots_dir = results_dir / "plots"
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

    # 4) Ablation study analysis
    if args.ablation:
        logger.info("Generating ablation study plots...")
        ablation_dir = results_dir / "ablation"
        if ablation_dir.exists():
            ablation_csvs = find_files(["aggregated_results_*.csv"], ablation_dir)
            if not ablation_csvs:
                logger.warning("No aggregated ablation results found.")
            for csv_path in ablation_csvs:
                plot_ablation_comparison(csv_path, plots_dir, metric=args.ablation_metric)
                plot_ablation_factors(csv_path, plots_dir)
        else:
            logger.warning(f"Ablation directory not found: {ablation_dir}")

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
