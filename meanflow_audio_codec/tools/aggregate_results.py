"""Aggregate evaluation results into table format.

This script reads evaluation CSVs and aggregates them into table format
with statistics (mean, std, min, max) for different groupings.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_results_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load results from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of result dictionaries
    """
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key, value in row.items():
                if key in ("mse", "psnr", "ssim", "pesq", "stoi", "spectral_distance", 
                          "inference_time_mean", "inference_time_std", 
                          "total_parameters", "total_parameters_millions", "nfe"):
                    try:
                        row[key] = float(value) if value else None
                    except (ValueError, TypeError):
                        row[key] = None
                elif key == "nfe":
                    try:
                        row[key] = int(value) if value else None
                    except (ValueError, TypeError):
                        row[key] = None
            rows.append(row)
    return rows


def compute_statistics(values: list[float | None]) -> dict[str, float | None]:
    """Compute statistics for a list of values.
    
    Args:
        values: List of numeric values (may contain None)
    
    Returns:
        Dictionary with mean, std, min, max
    """
    valid_values = [v for v in values if v is not None]
    
    if not valid_values:
        return {"mean": None, "std": None, "min": None, "max": None}
    
    arr = np.array(valid_values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_by_group(
    rows: list[dict[str, Any]],
    group_by: list[str],
    metrics: list[str],
) -> dict[tuple, dict[str, dict[str, float | None]]]:
    """Aggregate results by grouping keys.
    
    Args:
        rows: List of result rows
        group_by: List of column names to group by
        metrics: List of metric column names to aggregate
    
    Returns:
        Dictionary mapping (group_key_tuple) -> {metric: {stat: value}}
    """
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    
    # Group rows
    for row in rows:
        if row.get("status") != "success":
            continue
        
        group_key = tuple(str(row.get(key, "")) for key in group_by)
        grouped[group_key].append(row)
    
    # Compute statistics for each group
    aggregated: dict[tuple, dict[str, dict[str, float | None]]] = {}
    
    for group_key, group_rows in grouped.items():
        group_stats: dict[str, dict[str, float | None]] = {}
        
        for metric in metrics:
            values = [row.get(metric) for row in group_rows]
            group_stats[metric] = compute_statistics(values)
        
        aggregated[group_key] = group_stats
    
    return aggregated


def write_table_csv(
    aggregated: dict[tuple, dict[str, dict[str, float | None]]],
    group_by: list[str],
    metrics: list[str],
    output_path: Path,
) -> None:
    """Write aggregated results to CSV table.
    
    Args:
        aggregated: Aggregated results dictionary
        group_by: List of grouping column names
        metrics: List of metric column names
        output_path: Path to output CSV file
    """
    rows = []
    
    for group_key, stats in aggregated.items():
        row = {}
        
        # Add grouping columns
        for i, key in enumerate(group_by):
            row[key] = group_key[i] if i < len(group_key) else ""
        
        # Add metric statistics
        for metric in metrics:
            metric_stats = stats.get(metric, {})
            row[f"{metric}_mean"] = metric_stats.get("mean")
            row[f"{metric}_std"] = metric_stats.get("std")
            row[f"{metric}_min"] = metric_stats.get("min")
            row[f"{metric}_max"] = metric_stats.get("max")
        
        rows.append(row)
    
    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    """Main entry point for results aggregation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV file with evaluation results",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV file for aggregated results",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        default=["method", "architecture", "dataset", "tokenization", "nfe"],
        help="Columns to group by (default: method architecture dataset tokenization nfe)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mse", "psnr", "ssim", "inference_time_mean"],
        help="Metrics to aggregate (default: mse psnr ssim inference_time_mean)",
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input_csv}...")
    rows = load_results_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows")
    
    # Aggregate
    print(f"Aggregating by {args.group_by}...")
    aggregated = aggregate_by_group(rows, args.group_by, args.metrics)
    print(f"Found {len(aggregated)} groups")
    
    # Write table
    print(f"Writing aggregated table to {args.output_csv}...")
    write_table_csv(aggregated, args.group_by, args.metrics, args.output_csv)
    print("✓ Done")


if __name__ == "__main__":
    main()

