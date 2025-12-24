"""Evaluate all configs and aggregate results.

This script evaluates all configs in a directory, runs comprehensive evaluation,
and collects results into CSV/JSON format.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from meanflow_audio_codec.configs.config import load_config_from_json
from meanflow_audio_codec.datasets import audio, mnist
from meanflow_audio_codec.evaluators.comprehensive_evaluator import (
    ComprehensiveEvaluator,
)


def find_checkpoint(workdir: Path) -> Path | None:
    """Find latest checkpoint in workdir.
    
    Args:
        workdir: Work directory containing checkpoints
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = workdir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("step_*.msgpack"))
    if not checkpoints:
        return None
    
    # Sort by step number
    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (ValueError, IndexError):
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def load_real_data(config_path: Path, num_samples: int = 1000) -> np.ndarray:
    """Load real data for evaluation.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to load
    
    Returns:
        Real data samples, shape [N, ...]
    """
    config = load_config_from_json(config_path)
    dataset = config.dataset or "mnist"
    
    if dataset == "mnist":
        # Load MNIST test set
        data_iter = mnist.load_mnist(
            data_dir=config.data_dir or str(Path.home() / "datasets" / "mnist"),
            split="test",
            batch_size=min(100, num_samples),
            format="1d",
            normalize=True,
            seed=42,
        )
        
        samples = []
        for batch_images, _ in data_iter:
            samples.append(batch_images)
            if len(samples) * batch_images.shape[0] >= num_samples:
                break
        
        data = np.concatenate(samples, axis=0)[:num_samples]
        return data
    
    elif dataset == "audio":
        # Load audio data
        data_dir = config.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be provided for audio dataset")
        
        # Use audio pipeline to load samples
        # This is a simplified version - in practice, you'd want more control
        audio_iter = audio.build_audio_pipeline(
            data_dir=data_dir,
            seed=42,
            frame_sz=256 * 256 * 3,
            batch_size=min(32, num_samples),
        )
        
        samples = []
        for batch in audio_iter:
            # Flatten audio frames
            batch_flat = batch.reshape(batch.shape[0], -1)
            samples.append(batch_flat)
            if len(samples) * batch_flat.shape[0] >= num_samples:
                break
        
        data = np.concatenate(samples, axis=0)[:num_samples]
        return data
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def evaluate_config(
    config_path: Path,
    num_samples: int = 1000,
    n_steps_list: list[int] = [1, 10, 50, 250],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a single config.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to generate
        n_steps_list: List of NFE values to test
        output_dir: Optional output directory for results
    
    Returns:
        Evaluation results dictionary
    """
    config = load_config_from_json(config_path)
    
    # Find checkpoint
    workdir = Path(config.workdir)
    checkpoint = find_checkpoint(workdir)
    
    if checkpoint is None:
        return {
            "config_path": str(config_path),
            "status": "no_checkpoint",
            "error": "No checkpoint found",
        }
    
    try:
        # Load real data
        real_data = load_real_data(config_path, num_samples=num_samples)
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator(
            checkpoint_path=checkpoint,
            config_path=config_path,
            dataset=config.dataset or "mnist",
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            real_data=real_data,
            num_samples=num_samples,
            n_steps_list=n_steps_list,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        
        # Add metadata
        results["config_path"] = str(config_path)
        results["checkpoint_path"] = str(checkpoint)
        results["status"] = "success"
        
        # Save individual results if output_dir provided
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / f"{config_path.stem}_results.json"
            evaluator.save_results(results, results_file)
        
        return results
    
    except Exception as e:
        return {
            "config_path": str(config_path),
            "status": "error",
            "error": str(e),
        }


def aggregate_results_to_csv(
    results_list: list[dict[str, Any]], output_csv: Path
) -> None:
    """Aggregate evaluation results into CSV format.
    
    Args:
        results_list: List of evaluation result dictionaries
        output_csv: Path to output CSV file
    """
    rows = []
    
    for result in results_list:
        if result.get("status") != "success":
            # Add error row
            row = {
                "config_path": result.get("config_path", ""),
                "status": result.get("status", "unknown"),
                "error": result.get("error", ""),
            }
            rows.append(row)
            continue
        
        config = result.get("config", {})
        nfe_results = result.get("nfe_results", {})
        params = result.get("parameters", {})
        
        # Create a row for each NFE value
        for nfe_str, nfe_data in nfe_results.items():
            row = {
                "config_path": result.get("config_path", ""),
                "method": config.get("method", ""),
                "architecture": config.get("architecture", ""),
                "dataset": config.get("dataset", ""),
                "tokenization": config.get("tokenization", ""),
                "nfe": int(nfe_str),
                "mse": nfe_data.get("mse"),
                "psnr": nfe_data.get("psnr"),
                "ssim": nfe_data.get("ssim"),
                "pesq": nfe_data.get("pesq"),
                "stoi": nfe_data.get("stoi"),
                "spectral_distance": nfe_data.get("spectral_distance"),
                "inference_time_mean": nfe_data.get("inference_time", {}).get("mean"),
                "inference_time_std": nfe_data.get("inference_time", {}).get("std"),
                "total_parameters": params.get("total"),
                "total_parameters_millions": params.get("total_millions"),
            }
            rows.append(row)
    
    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    """Main entry point for evaluation runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all configs")
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing config files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory for output results (default: evaluation_results)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluation_results.csv"),
        help="Path to output CSV file (default: evaluation_results.csv)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        nargs="+",
        default=[1, 10, 50, 250],
        help="NFE values to test (default: 1 10 50 250)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel evaluations (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Find all config files
    config_files = list(args.config_dir.glob("*.json"))
    print(f"Found {len(config_files)} config files")
    
    # Evaluate each config
    results = []
    for i, config_path in enumerate(config_files):
        print(f"\n[{i+1}/{len(config_files)}] Evaluating {config_path.name}...")
        result = evaluate_config(
            config_path=config_path,
            num_samples=args.num_samples,
            n_steps_list=args.n_steps,
            output_dir=args.output_dir,
        )
        results.append(result)
        
        if result.get("status") == "success":
            print(f"  ✓ Success")
        else:
            print(f"  ✗ {result.get('status')}: {result.get('error', 'Unknown error')}")
    
    # Aggregate results
    print(f"\nAggregating results to {args.output_csv}...")
    aggregate_results_to_csv(results, args.output_csv)
    print(f"✓ Done. Wrote {len(results)} results to CSV")


if __name__ == "__main__":
    main()

