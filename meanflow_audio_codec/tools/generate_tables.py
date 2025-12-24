"""Generate experimental tables from evaluation results.

This module provides functions to generate all 8 experimental tables
in multiple formats (CSV, LaTeX, Markdown, HTML).
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from meanflow_audio_codec.tools.aggregate_results import load_results_csv


def load_aggregated_results(csv_path: Path) -> list[dict[str, Any]]:
    """Load aggregated results from CSV.
    
    Args:
        csv_path: Path to aggregated results CSV
    
    Returns:
        List of result dictionaries
    """
    return load_results_csv(csv_path)


def format_value(value: float | None, fmt: str = ".3f") -> str:
    """Format a numeric value for table display.
    
    Args:
        value: Numeric value (may be None)
        fmt: Format string
    
    Returns:
        Formatted string
    """
    if value is None:
        return "—"
    return f"{value:{fmt}}"


def generate_main_results_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 1: Main Results - Comprehensive Method Comparison.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format ("latex", "markdown", "csv", "html")
    """
    # Filter to best NFE for each method/architecture/dataset/tokenization
    # Group by method, architecture, dataset, tokenization
    grouped: dict[tuple, dict[str, Any]] = {}
    
    for row in results:
        if row.get("status") != "success":
            continue
        
        key = (
            row.get("method", ""),
            row.get("architecture", ""),
            row.get("dataset", ""),
            row.get("tokenization", ""),
        )
        
        nfe = row.get("nfe", 0)
        
        if key not in grouped or nfe < grouped[key].get("nfe", float("inf")):
            grouped[key] = row
    
    # Generate table rows
    rows = []
    for key, row in sorted(grouped.items()):
        rows.append({
            "Method": key[0] or "—",
            "Architecture": key[1] or "—",
            "Dataset": key[2] or "—",
            "Tokenization": key[3] or "—",
            "FID": format_value(row.get("fid")),
            "KID": format_value(row.get("kid")),
            "MSE": format_value(row.get("mse")),
            "PSNR": format_value(row.get("psnr")),
            "SSIM": format_value(row.get("ssim")),
            "NFE": int(row.get("nfe", 0)) if row.get("nfe") else "—",
            "Training Time (hrs)": format_value(row.get("training_time_hours")),
            "Params (M)": format_value(row.get("total_parameters_millions"), ".2f"),
        })
    
    _write_table(rows, output_path, format, "Main Results - Comprehensive Method Comparison")


def generate_method_ablation_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 2: Method Ablation Study.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # Filter to method ablation results
    method_rows = [
        r for r in results
        if r.get("status") == "success" and "no_stop_gradient" in str(r.get("config_path", ""))
        or "time_dependent" in str(r.get("config_path", ""))
        or "learned" in str(r.get("config_path", ""))
    ]
    
    rows = []
    for row in method_rows:
        config_path = str(row.get("config_path", ""))
        variant = "baseline"
        if "no_stop_gradient" in config_path:
            variant = "w/o stop-grad"
        elif "time_dependent" in config_path:
            variant = "time-dependent loss"
        elif "learned" in config_path:
            variant = "learned loss"
        
        rows.append({
            "Method Variant": variant,
            "Description": _get_variant_description(variant),
            "FID": format_value(row.get("fid")),
            "KID": format_value(row.get("kid")),
            "Training Stability": "—",  # Would need additional metrics
        })
    
    _write_table(rows, output_path, format, "Method Ablation Study")


def generate_architecture_ablation_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 3: Architecture Ablation Study.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # Group by architecture and scale
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    
    for row in results:
        if row.get("status") != "success":
            continue
        
        config_path = str(row.get("config_path", ""))
        architecture = row.get("architecture", "")
        scale = "medium"
        if "scale=small" in config_path:
            scale = "small"
        elif "scale=large" in config_path:
            scale = "large"
        
        key = (architecture, scale)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    
    rows = []
    for (arch, scale), group_rows in sorted(grouped.items()):
        # Get representative row (best performance or median)
        rep_row = group_rows[0]  # Simplified: take first
        
        rows.append({
            "Architecture": arch or "—",
            "Scale": scale,
            "Hidden Dim": _get_hidden_dim(arch, scale),
            "Num Blocks": _get_num_blocks(arch, scale),
            "Params (M)": format_value(rep_row.get("total_parameters_millions"), ".2f"),
            "FID": format_value(rep_row.get("fid")),
            "Training Speed": format_value(rep_row.get("inference_time_mean")),  # Proxy
            "Memory": "—",  # Would need memory data
        })
    
    _write_table(rows, output_path, format, "Architecture Ablation Study")


def generate_tokenization_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 4: Tokenization Strategy Comparison.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # Group by method, architecture, tokenization
    grouped: dict[tuple, dict[str, Any]] = {}
    
    for row in results:
        if row.get("status") != "success":
            continue
        
        key = (
            row.get("method", ""),
            row.get("architecture", ""),
            row.get("tokenization", ""),
        )
        
        # Take best NFE
        nfe = row.get("nfe", 0)
        if key not in grouped or nfe < grouped[key].get("nfe", float("inf")):
            grouped[key] = row
    
    rows = []
    for key, row in sorted(grouped.items()):
        rows.append({
            "Method": key[0] or "—",
            "Architecture": key[1] or "—",
            "Tokenization": key[2] or "—",
            "FID": format_value(row.get("fid")),
            "Reconstruction Quality": format_value(row.get("psnr")),  # Proxy
            "Training Stability": "—",
            "Token Efficiency": "—",
        })
    
    _write_table(rows, output_path, format, "Tokenization Strategy Comparison")


def generate_hyperparameter_sensitivity_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 5: Hyperparameter Sensitivity.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # Extract hyperparameter sweep results
    hyperparams = ["gamma", "flow_ratio", "base_lr"]
    
    rows = []
    for hyperparam in hyperparams:
        # Find rows with this hyperparameter variation
        param_rows = [
            r for r in results
            if f"{hyperparam}=" in str(r.get("config_path", ""))
            and r.get("status") == "success"
        ]
        
        if not param_rows:
            continue
        
        # Extract values and metrics
        values = []
        fids = []
        
        for row in param_rows:
            config_path = str(row.get("config_path", ""))
            # Extract value from config path
            for part in config_path.split("--"):
                if part.startswith(f"{hyperparam}="):
                    try:
                        value = float(part.split("=")[1])
                        values.append(value)
                        fid = row.get("fid")
                        if fid is not None:
                            fids.append(fid)
                    except (ValueError, IndexError):
                        pass
        
        if values and fids:
            min_fid = min(fids)
            max_fid = max(fids)
            best_value = values[fids.index(min_fid)]
            
            rows.append({
                "Hyperparameter": hyperparam,
                "Value Range": f"{min(values):.4f} - {max(values):.4f}",
                "Best Value": format_value(best_value),
                "FID (min)": format_value(min_fid),
                "FID (max)": format_value(max_fid),
                "Sensitivity": "High" if (max_fid - min_fid) > 1.0 else "Low",
            })
    
    _write_table(rows, output_path, format, "Hyperparameter Sensitivity")


def generate_efficiency_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 6: Computational Efficiency.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # Group by method, architecture, NFE
    grouped: dict[tuple, dict[str, Any]] = {}
    
    for row in results:
        if row.get("status") != "success":
            continue
        
        key = (
            row.get("method", ""),
            row.get("architecture", ""),
            row.get("nfe", 0),
        )
        
        if key not in grouped:
            grouped[key] = row
    
    rows = []
    for key, row in sorted(grouped.items()):
        nfe = key[2]
        inference_time = row.get("inference_time_mean", 0)
        speedup = 250.0 / nfe if nfe > 0 else 0  # Relative to 250-step flow matching
        
        rows.append({
            "Method": key[0] or "—",
            "Architecture": key[1] or "—",
            "NFE": int(nfe) if nfe else "—",
            "Inference Time (ms)": format_value(inference_time * 1000, ".2f"),
            "Training Time (hrs)": format_value(row.get("training_time_hours")),
            "Memory (GB)": "—",  # Would need memory data
            "FID": format_value(row.get("fid")),
            "Speedup vs Flow Matching": format_value(speedup, ".1f") + "×",
        })
    
    _write_table(rows, output_path, format, "Computational Efficiency")


def generate_dataset_transfer_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 7: Dataset Transfer (MNIST → Audio).
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # Group by method, architecture, dataset
    grouped: dict[tuple, dict[str, Any]] = {}
    
    for row in results:
        if row.get("status") != "success":
            continue
        
        key = (
            row.get("method", ""),
            row.get("architecture", ""),
            row.get("dataset", ""),
        )
        
        # Take best NFE
        nfe = row.get("nfe", 0)
        if key not in grouped or nfe < grouped[key].get("nfe", float("inf")):
            grouped[key] = row
    
    rows = []
    for (method, arch, dataset), row in sorted(grouped.items()):
        rows.append({
            "Method": method or "—",
            "Architecture": arch or "—",
            "Dataset": dataset or "—",
            "FID": format_value(row.get("fid")),
            "Domain-Specific Metric": format_value(row.get("psnr") or row.get("pesq")),
            "Transfer Success": "—",  # Would need comparison logic
        })
    
    _write_table(rows, output_path, format, "Dataset Transfer (MNIST → Audio)")


def generate_baseline_comparison_table(
    results: list[dict[str, Any]], output_path: Path, format: str = "latex"
) -> None:
    """Generate Table 8: Comparison with Baselines.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format
    """
    # This table would compare against published baselines
    # For now, we'll show our results with placeholder baseline columns
    rows = []
    
    # Get best results for each method
    method_best: dict[str, dict[str, Any]] = {}
    
    for row in results:
        if row.get("status") != "success":
            continue
        
        method = row.get("method", "")
        if method and (method not in method_best or row.get("fid", float("inf")) < method_best[method].get("fid", float("inf"))):
            method_best[method] = row
    
    for method, row in sorted(method_best.items()):
        rows.append({
            "Method": method or "—",
            "Architecture": row.get("architecture", "—"),
            "Dataset": row.get("dataset", "—"),
            "Our FID": format_value(row.get("fid")),
            "Baseline FID": "—",  # Would need baseline data
            "Reference": "—",  # Would need reference citations
            "Improvement": "—",
        })
    
    _write_table(rows, output_path, format, "Comparison with Baselines")


def _write_table(
    rows: list[dict[str, Any]], output_path: Path, format: str, title: str
) -> None:
    """Write table in specified format.
    
    Args:
        rows: List of row dictionaries
        output_path: Path to output file
        format: Output format ("latex", "markdown", "csv", "html")
        title: Table title
    """
    if not rows:
        return
    
    fieldnames = list(rows[0].keys())
    
    if format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    elif format == "latex":
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{table}\n")
            f.write(f"\\caption{{{title}}}\n")
            f.write("\\begin{tabular}{" + "l" * len(fieldnames) + "}\n")
            f.write("\\toprule\n")
            f.write(" & ".join(fieldnames) + " \\\\\n")
            f.write("\\midrule\n")
            for row in rows:
                f.write(" & ".join(str(row.get(f, "—")) for f in fieldnames) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    elif format == "markdown":
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write("| " + " | ".join(fieldnames) + " |\n")
            f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(f, "—")) for f in fieldnames) + " |\n")
    
    elif format == "html":
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"<h2>{title}</h2>\n")
            f.write("<table>\n")
            f.write("<thead><tr>")
            for field in fieldnames:
                f.write(f"<th>{field}</th>")
            f.write("</tr></thead>\n")
            f.write("<tbody>\n")
            for row in rows:
                f.write("<tr>")
                for field in fieldnames:
                    f.write(f"<td>{row.get(field, '—')}</td>")
                f.write("</tr>\n")
            f.write("</tbody>\n")
            f.write("</table>\n")


def _get_variant_description(variant: str) -> str:
    """Get description for method variant."""
    descriptions = {
        "baseline": "Standard implementation",
        "w/o stop-grad": "Mean Flow without stop-gradient on target",
        "time-dependent loss": "Time-dependent loss weighting",
        "learned loss": "Learned loss weighting",
    }
    return descriptions.get(variant, "")


def _get_hidden_dim(arch: str, scale: str) -> str:
    """Get hidden dimension for architecture/scale."""
    dims = {
        ("mlp", "small"): "256",
        ("mlp", "medium"): "512",
        ("mlp", "large"): "1024",
        ("mlp_mixer", "small"): "256",
        ("mlp_mixer", "medium"): "512",
        ("mlp_mixer", "large"): "1024",
        ("convnet", "small"): "256",
        ("convnet", "medium"): "512",
        ("convnet", "large"): "1024",
    }
    return dims.get((arch, scale), "—")


def _get_num_blocks(arch: str, scale: str) -> str:
    """Get number of blocks for architecture/scale."""
    blocks = {
        ("mlp", "small"): "4",
        ("mlp", "medium"): "8",
        ("mlp", "large"): "16",
        ("mlp_mixer", "small"): "4",
        ("mlp_mixer", "medium"): "8",
        ("mlp_mixer", "large"): "16",
        ("convnet", "small"): "4",
        ("convnet", "medium"): "8",
        ("convnet", "large"): "16",
    }
    return blocks.get((arch, scale), "—")


def main():
    """Main entry point for table generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experimental tables")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV file with aggregated results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Output directory for tables (default: tables)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["latex", "markdown", "csv", "html"],
        default="latex",
        help="Output format (default: latex)",
    )
    parser.add_argument(
        "--table",
        type=int,
        choices=range(1, 9),
        help="Generate specific table (1-8), or all if not specified",
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input_csv}...")
    results = load_aggregated_results(args.input_csv)
    print(f"Loaded {len(results)} result rows")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table generators
    generators = [
        (1, generate_main_results_table, "main_results"),
        (2, generate_method_ablation_table, "method_ablation"),
        (3, generate_architecture_ablation_table, "architecture_ablation"),
        (4, generate_tokenization_table, "tokenization"),
        (5, generate_hyperparameter_sensitivity_table, "hyperparameter_sensitivity"),
        (6, generate_efficiency_table, "efficiency"),
        (7, generate_dataset_transfer_table, "dataset_transfer"),
        (8, generate_baseline_comparison_table, "baseline_comparison"),
    ]
    
    # Generate tables
    for table_num, generator, name in generators:
        if args.table is None or args.table == table_num:
            ext = {"latex": "tex", "markdown": "md", "csv": "csv", "html": "html"}[args.format]
            output_path = args.output_dir / f"table_{table_num}_{name}.{ext}"
            print(f"Generating Table {table_num}: {name}...")
            generator(results, output_path, args.format)
            print(f"  ✓ Wrote to {output_path}")
    
    print("\n✓ Done")


if __name__ == "__main__":
    main()

