"""Generate configuration files for all experimental combinations.

This script generates:
1. Base configs: 4 methods × 3 architectures × 2 tokenizations × 2 datasets = 48 configs
2. Ablation configs: hyperparameter sweeps, architecture scaling, method variants
"""

import json
from pathlib import Path
from typing import Any

# Base configuration template
BASE_CONFIG_TEMPLATE = {
    "batch_size": 128,
    "n_steps": 5000,
    "sample_every": 500,
    "sample_seed": 42,
    "sample_steps": 50,
    "base_lr": 0.0001,
    "weight_decay": 0.0001,
    "seed": 42,
    "checkpoint_step": 200,
    "data_dir": None,
    "noise_dimension": 784,  # Will be overridden for audio
    "condition_dimension": 128,
    "latent_dimension": 256,
    "num_blocks": 8,
}

# Method-specific defaults
METHOD_DEFAULTS = {
    "autoencoder": {
        "use_improved_mean_flow": False,
        "method": "autoencoder",
    },
    "flow_matching": {
        "use_improved_mean_flow": False,
        "method": "flow_matching",
    },
    "mean_flow": {
        "use_improved_mean_flow": False,
        "method": "mean_flow",
        "gamma": 1.0,
        "flow_ratio": 1.0,
        "c": 1.0,
        "use_stop_gradient": True,
    },
    "improved_mean_flow": {
        "use_improved_mean_flow": True,
        "method": "improved_mean_flow",
        "gamma": 1.0,
        "flow_ratio": 1.0,
        "c": 1.0,
        "use_stop_gradient": True,
    },
}

# Architecture-specific defaults
ARCHITECTURE_DEFAULTS = {
    "mlp": {
        "architecture": "mlp",
    },
    "mlp_mixer": {
        "architecture": "mlp_mixer",
    },
    "convnet": {
        "architecture": "convnet",
    },
}

# Dataset-specific defaults
DATASET_DEFAULTS = {
    "mnist": {
        "dataset": "mnist",
        "noise_dimension": 784,  # 28*28
    },
    "audio": {
        "dataset": "audio",
        "noise_dimension": 256 * 256 * 3,  # frame_sz * n_channels
    },
}

# Tokenization configs
TOKENIZATION_CONFIGS = {
    "mdct": {
        "tokenization_strategy": "mdct",
        "tokenization_config": {
            "window_size": 512,
            "hop_size": 256,
        },
    },
    "reshape": {
        "tokenization_strategy": "reshape",
        "tokenization_config": {
            "patch_size": 4,  # For MNIST (28x28, 4x4 patches = 7x7 = 49 patches)
            "patch_length": 128,  # For audio
        },
    },
}


def generate_base_configs(output_dir: Path) -> list[Path]:
    """Generate all 48 base configs.
    
    Args:
        output_dir: Directory to write config files
    
    Returns:
        List of generated config file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    methods = ["autoencoder", "flow_matching", "mean_flow", "improved_mean_flow"]
    architectures = ["mlp", "mlp_mixer", "convnet"]
    tokenizations = ["mdct", "reshape"]
    datasets = ["mnist", "audio"]
    
    for method in methods:
        for architecture in architectures:
            for tokenization in tokenizations:
                for dataset in datasets:
                    # Build config
                    config = BASE_CONFIG_TEMPLATE.copy()
                    config.update(METHOD_DEFAULTS[method])
                    config.update(ARCHITECTURE_DEFAULTS[architecture])
                    config.update(DATASET_DEFAULTS[dataset])
                    config.update(TOKENIZATION_CONFIGS[tokenization])
                    
                    # Set workdir (relative to project root)
                    config["workdir"] = f"./outputs/method={method}--architecture={architecture}--dataset={dataset}--tokenization={tokenization}"
                    
                    # Generate filename
                    filename = (
                        f"method={method}--architecture={architecture}--dataset={dataset}--tokenization={tokenization}.json"
                    )
                    filepath = output_dir / filename
                    
                    # Write config
                    with filepath.open("w", encoding="utf-8") as f:
                        json.dump(config, f, indent=2, sort_keys=True)
                    
                    generated_files.append(filepath)
    
    return generated_files


def generate_ablation_configs(
    base_config_dir: Path,
    output_dir: Path,
    ablation_type: str,
    values: list[Any],
    param_name: str,
) -> list[Path]:
    """Generate ablation configs by varying a single parameter.
    
    Args:
        base_config_dir: Directory containing base configs
        output_dir: Directory to write ablation configs
        ablation_type: Type of ablation (e.g., "gamma", "flow_ratio", "lr")
        values: List of values to sweep
        param_name: Name of parameter to vary
    
    Returns:
        List of generated config file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # Find all base configs
    base_configs = list(base_config_dir.glob("method=*.json"))
    
    for base_config_path in base_configs:
        with base_config_path.open("r", encoding="utf-8") as f:
            base_config = json.load(f)
        
        # Extract method, architecture, dataset, tokenization from filename
        stem = base_config_path.stem
        parts = stem.split("--")
        method = None
        architecture = None
        dataset = None
        tokenization = None
        
        for part in parts:
            if part.startswith("method="):
                method = part.split("=")[1]
            elif part.startswith("architecture="):
                architecture = part.split("=")[1]
            elif part.startswith("dataset="):
                dataset = part.split("=")[1]
            elif part.startswith("tokenization="):
                tokenization = part.split("=")[1]
        
        # Generate configs for each value
        for value in values:
            config = base_config.copy()
            config[param_name] = value
            
            # Update workdir
            base_workdir = config.get("workdir", "")
            config["workdir"] = f"{base_workdir}--{ablation_type}={value}"
            
            # Generate filename
            filename = f"{stem}--{ablation_type}={value}.json"
            filepath = output_dir / filename
            
            # Write config
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            generated_files.append(filepath)
    
    return generated_files


def generate_architecture_scaling_configs(
    base_config_dir: Path, output_dir: Path
) -> list[Path]:
    """Generate architecture scaling configs (small/medium/large).
    
    Args:
        base_config_dir: Directory containing base configs
        output_dir: Directory to write scaling configs
    
    Returns:
        List of generated config file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # Architecture scaling definitions
    scaling_configs = {
        "small": {
            "num_blocks": 4,
            "latent_dimension": 256,
            "condition_dimension": 128,
        },
        "medium": {
            "num_blocks": 8,
            "latent_dimension": 512,
            "condition_dimension": 256,
        },
        "large": {
            "num_blocks": 16,
            "latent_dimension": 1024,
            "condition_dimension": 512,
        },
    }
    
    # Find all base configs
    base_configs = list(base_config_dir.glob("method=*.json"))
    
    for base_config_path in base_configs:
        with base_config_path.open("r", encoding="utf-8") as f:
            base_config = json.load(f)
        
        stem = base_config_path.stem
        
        for scale_name, scale_config in scaling_configs.items():
            config = base_config.copy()
            config.update(scale_config)
            
            # Update workdir
            base_workdir = config.get("workdir", "")
            config["workdir"] = f"{base_workdir}--scale={scale_name}"
            
            # Generate filename
            filename = f"{stem}--scale={scale_name}.json"
            filepath = output_dir / filename
            
            # Write config
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            generated_files.append(filepath)
    
    return generated_files


def generate_method_ablation_configs(
    base_config_dir: Path, output_dir: Path
) -> list[Path]:
    """Generate method ablation configs (stop-gradient, loss weighting, etc.).
    
    Args:
        base_config_dir: Directory containing base configs
        output_dir: Directory to write ablation configs
    
    Returns:
        List of generated config file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # Method ablations
    ablations = [
        {"use_stop_gradient": False, "name": "no_stop_gradient"},
        {"loss_weighting": "time_dependent", "name": "time_dependent_loss"},
        {"loss_weighting": "learned", "name": "learned_loss"},
    ]
    
    # Find all base configs that use mean_flow or improved_mean_flow
    base_configs = list(base_config_dir.glob("method=mean_flow*.json"))
    base_configs.extend(base_config_dir.glob("method=improved_mean_flow*.json"))
    
    for base_config_path in base_configs:
        with base_config_path.open("r", encoding="utf-8") as f:
            base_config = json.load(f)
        
        stem = base_config_path.stem
        
        for ablation in ablations:
            config = base_config.copy()
            ablation_params = {k: v for k, v in ablation.items() if k != "name"}
            config.update(ablation_params)
            
            # Update workdir
            base_workdir = config.get("workdir", "")
            config["workdir"] = f"{base_workdir}--{ablation['name']}"
            
            # Generate filename
            filename = f"{stem}--{ablation['name']}.json"
            filepath = output_dir / filename
            
            # Write config
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            generated_files.append(filepath)
    
    return generated_files


def main():
    """Main entry point for config generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experimental configs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs"),
        help="Directory to write configs (default: configs)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only generate base configs (48 configs)",
    )
    parser.add_argument(
        "--ablations-dir",
        type=Path,
        default=Path("configs/ablations"),
        help="Directory for ablation configs (default: configs/ablations)",
    )
    
    args = parser.parse_args()
    
    # Generate base configs
    print("Generating base configs...")
    base_files = generate_base_configs(args.output_dir)
    print(f"Generated {len(base_files)} base configs")
    
    if args.base_only:
        return
    
    # Generate ablation configs
    print("\nGenerating ablation configs...")
    
    # Hyperparameter sweeps
    print("  - Gamma sweep...")
    gamma_files = generate_ablation_configs(
        args.output_dir,
        args.ablations_dir / "gamma_sweep",
        "gamma",
        [0.0, 0.5, 1.0, 1.5, 2.0],
        "gamma",
    )
    print(f"    Generated {len(gamma_files)} configs")
    
    print("  - Flow ratio sweep...")
    flow_ratio_files = generate_ablation_configs(
        args.output_dir,
        args.ablations_dir / "flow_ratio_sweep",
        "flow_ratio",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        "flow_ratio",
    )
    print(f"    Generated {len(flow_ratio_files)} configs")
    
    print("  - Learning rate sweep...")
    lr_files = generate_ablation_configs(
        args.output_dir,
        args.ablations_dir / "lr_sweep",
        "lr",
        [1e-5, 5e-5, 1e-4, 5e-4],
        "base_lr",
    )
    print(f"    Generated {len(lr_files)} configs")
    
    # Architecture scaling
    print("  - Architecture scaling...")
    scaling_files = generate_architecture_scaling_configs(
        args.output_dir,
        args.ablations_dir / "architecture_scaling",
    )
    print(f"    Generated {len(scaling_files)} configs")
    
    # Method ablations
    print("  - Method ablations...")
    method_files = generate_method_ablation_configs(
        args.output_dir,
        args.ablations_dir / "method_ablations",
    )
    print(f"    Generated {len(method_files)} configs")
    
    print(f"\nTotal ablation configs: {len(gamma_files) + len(flow_ratio_files) + len(lr_files) + len(scaling_files) + len(method_files)}")


if __name__ == "__main__":
    main()

