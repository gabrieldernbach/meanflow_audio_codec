#!/usr/bin/env python3
"""Main training entry point.

Provides CLI interface for training flow models.
Supports config file loading, workdir management, and checkpoint resumption.
"""

import argparse
from pathlib import Path

from meanflow_audio_codec.configs.config import TrainFlowConfig, load_config_from_json
from meanflow_audio_codec.trainers.train import train_flow


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train flow models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common arguments
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file (overrides other arguments)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Working directory for outputs (samples, checkpoints, logs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in workdir",
    )

    # Flow model arguments (used if --config not provided)
    flow_group = parser.add_argument_group("Flow model arguments")
    flow_group.add_argument("--batch-size", type=int, help="Batch size")
    flow_group.add_argument("--n-steps", type=int, help="Number of training steps")
    flow_group.add_argument(
        "--sample-every", type=int, help="Sample every N steps"
    )
    flow_group.add_argument("--sample-seed", type=int, help="Random seed for sampling")
    flow_group.add_argument(
        "--sample-steps", type=int, help="Number of steps for sampling"
    )
    flow_group.add_argument("--base-lr", type=float, help="Base learning rate")
    flow_group.add_argument("--weight-decay", type=float, help="Weight decay")
    flow_group.add_argument("--seed", type=int, help="Random seed")
    flow_group.add_argument(
        "--use-improved-mean-flow",
        action="store_true",
        help="Use improved mean flow objective",
    )
    flow_group.add_argument(
        "--checkpoint-step", type=int, default=None, help="Step to save checkpoint"
    )
    flow_group.add_argument("--data-dir", type=str, help="Data directory")
    flow_group.add_argument("--noise-dimension", type=int, help="Noise dimension")
    flow_group.add_argument(
        "--condition-dimension", type=int, help="Condition dimension"
    )
    flow_group.add_argument("--latent-dimension", type=int, help="Latent dimension")
    flow_group.add_argument("--num-blocks", type=int, help="Number of blocks")

    args = parser.parse_args()

    # Load config from file or create from arguments
    if args.config:
        config = load_config_from_json(args.config)
        # Override workdir from CLI if provided
        if args.workdir:
            config.workdir = args.workdir
    else:
        # Validate required flow arguments
        required_flow_args = [
            "batch_size",
            "n_steps",
            "sample_every",
            "sample_seed",
            "sample_steps",
            "base_lr",
            "weight_decay",
            "seed",
            "noise_dimension",
            "condition_dimension",
            "latent_dimension",
            "num_blocks",
        ]
        missing = [
            arg
            for arg in required_flow_args
            if getattr(args, arg.replace("-", "_"), None) is None
        ]
        if missing:
            parser.error(
                f"Missing required flow arguments: {', '.join(missing)}. "
                "Either provide --config or all required arguments."
            )

        config = TrainFlowConfig(
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            sample_seed=args.sample_seed,
            sample_steps=args.sample_steps,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            use_improved_mean_flow=args.use_improved_mean_flow,
            checkpoint_step=args.checkpoint_step,
            data_dir=args.data_dir,
            noise_dimension=args.noise_dimension,
            condition_dimension=args.condition_dimension,
            latent_dimension=args.latent_dimension,
            num_blocks=args.num_blocks,
            workdir=args.workdir,
        )

    # Train flow model
    train_flow(config, resume=args.resume)


if __name__ == "__main__":
    main()
