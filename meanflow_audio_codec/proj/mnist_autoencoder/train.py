#!/usr/bin/env python3
"""Train MLPMixerAutoencoder on MNIST with tiny config for development.

Project-specific training script for MNIST autoencoder.
Method (A) Pure Autoencoder on MNIST dataset.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from meanflow_audio_codec.datasets import load_mnist
from meanflow_audio_codec.models import MLPMixerAutoencoder, TrainState
from meanflow_audio_codec.trainers.utils import (
    LogWriter,
    find_latest_checkpoint,
    load_checkpoint_and_resume,
    plot_samples,
    save_checkpoint_with_metadata,
)
from meanflow_audio_codec.utils import ema


@jax.jit
def train_step_autoencoder(state: TrainState, x: jnp.ndarray) -> tuple[TrainState, jnp.ndarray]:
    """Autoencoder training step with reconstruction loss."""
    def loss_fn(params):
        recon = state.apply_fn({"params": params}, x)
        loss = jnp.mean((recon - x) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    # Training configuration - appropriate for M1 MacBook 16GB RAM
    batch_size = 128  # Smaller batch for memory constraints
    n_steps = 2000
    sample_every = 200
    base_lr = 1e-3  # Higher learning rate for autoencoder
    weight_decay = 1e-4
    seed = 42
    resume = False

    # Model hyperparameters - TINY config for development
    input_dim = 784  # 28x28 for MNIST
    num_latent_tokens = 16  # Small number of latent tokens
    latent_dim = 64  # Small latent dimension
    num_context_tokens = 128  # Reduced from default 512
    num_output_tokens = 128  # Reduced from default 512
    token_mix_dim = 256  # Reduced from default 2048
    channel_mix_dim = 256  # Reduced from default 2048

    # Setup workdir
    workdir = Path("./outputs/trial_mlp_mixer_autoencoder")
    samples_dir = workdir / "samples"
    checkpoints_dir = workdir / "checkpoints"
    logs_dir = workdir / "logs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Write initial info to log
    stdout_log = logs_dir / "stdout.log"
    stderr_log = logs_dir / "stderr.log"
    stdout_file = open(stdout_log, "w", buffering=1)
    stderr_file = open(stderr_log, "w", buffering=1)

    header = (
        "=" * 60 + "\n"
        "TRAINING: MLPMixerAutoencoder on MNIST\n"
        "=" * 60 + "\n"
        f"Steps: {n_steps}, Batch size: {batch_size}\n"
        f"Model: input_dim={input_dim}, num_latent_tokens={num_latent_tokens}\n"
        f"latent_dim={latent_dim}, num_context_tokens={num_context_tokens}\n"
        f"num_output_tokens={num_output_tokens}\n"
        f"token_mix_dim={token_mix_dim}, channel_mix_dim={channel_mix_dim}\n"
        f"Learning rate: {base_lr}\n"
        f"Workdir: {workdir}\n"
        "-" * 60 + "\n"
    )
    print(header)
    stdout_file.write(header)
    stdout_file.flush()

    # Create model
    print("Initializing model...")
    model = MLPMixerAutoencoder(
        input_dim=input_dim,
        num_latent_tokens=num_latent_tokens,
        latent_dim=latent_dim,
        num_context_tokens=num_context_tokens,
        num_output_tokens=num_output_tokens,
        token_mix_dim=token_mix_dim,
        channel_mix_dim=channel_mix_dim,
    )

    # Setup optimizer
    tx = optax.adamw(learning_rate=base_lr, weight_decay=weight_decay)

    # Initialize model
    print("Initializing parameters...")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {[str(d) for d in jax.devices()]}")

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)

    x0 = jnp.zeros((batch_size, model.input_dim), dtype=jnp.float32)
    params = model.init(k_init, x0)["params"]

    # Count parameters
    from jax.tree_util import tree_map, tree_reduce

    param_counts = tree_map(lambda x: x.size if hasattr(x, "size") else 0, params)
    total_params = tree_reduce(lambda a, b: a + b, param_counts, 0)
    print(f"✓ Model initialized with {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # Verify computation runs
    test_output = model.apply({"params": params}, x0)
    print(f"✓ Test forward pass on device: {test_output.device}")

    state_template = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Handle checkpoint resume
    start_step = 0
    if resume:
        checkpoint_path = find_latest_checkpoint(workdir)
        if checkpoint_path is not None and checkpoint_path.stat().st_size > 0:
            print(f"Found checkpoint: {checkpoint_path} ({checkpoint_path.stat().st_size / 1e6:.1f}MB)")
            try:
                state_template, start_step = load_checkpoint_and_resume(workdir, state_template, config=None)
                print(f"✓ Resuming from checkpoint at step {start_step}")
            except Exception as e:
                print(f"⚠ Failed to load checkpoint: {e}")
                print("Starting from scratch instead...")
                start_step = 0
        else:
            print("No valid checkpoint found, starting from scratch")

    state = state_template

    # Load data using existing MNIST dataloader
    data_dir = str(Path.home() / "datasets" / "mnist")
    print("Loading MNIST dataset...")
    it = load_mnist(
        data_dir=data_dir,
        split="train",
        batch_size=batch_size,
        format="1d",
        normalize=True,
        seed=seed,
    )

    # Setup logging
    log_path = logs_dir / "train_log.jsonl"
    logger = LogWriter(log_path)

    loss_avg = None

    training_start = (
        "\nStarting training...\n"
        f"{'Step':<6} {'Loss':<12} {'Avg Loss':<12} {'Status'}\n"
        "-" * 60 + "\n"
    )
    print(training_start)
    stdout_file.write(training_start)
    stdout_file.flush()

    for step in range(start_step, n_steps):
        x, _ = next(it)
        x = jnp.asarray(x)

        state, loss = train_step_autoencoder(state, x)

        loss_val = float(loss)
        loss_avg = ema(loss_avg, loss_val)

        # Log to JSON
        logger.write_step(
            step,
            {
                "loss": loss_val,
                "loss_avg": loss_avg if loss_avg is not None else loss_val,
                "lr": base_lr,
            },
        )

        # Print progress
        if step % 25 == 0:
            msg = f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f}"
            print(msg)
            stdout_file.write(msg + "\n")
            stdout_file.flush()

        # Sample periodically
        if step % sample_every == 0:
            msg = f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f} sampling..."
            print(msg)
            stdout_file.write(msg + "\n")
            stdout_file.flush()

            # Generate reconstructions
            with jax.disable_jit():
                recon = model.apply({"params": state.params}, x[:16])
                recon_np = np.array(recon).reshape(16, 28, 28)
                orig_np = np.array(x[:16]).reshape(16, 28, 28)

                # Plot original and reconstructed side by side
                plot_samples(
                    np.concatenate([orig_np, recon_np], axis=0),
                    np.concatenate([np.zeros(16, dtype=np.int32), np.ones(16, dtype=np.int32)]),
                    samples_dir / f"step_{step:04d}.png",
                )

            # Save checkpoint
            checkpoint_path = checkpoints_dir / f"step_{step:05d}.msgpack"
            save_checkpoint_with_metadata(checkpoint_path, state, step, config=None)
            latest_path = checkpoints_dir / "latest.msgpack"
            save_checkpoint_with_metadata(latest_path, state, step, config=None)
            msg = f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f} ✓ saved"
            print(msg)
            stdout_file.write(msg + "\n")
            stdout_file.flush()

    # Final sample
    final_sample_msg = "-" * 60 + "\nGenerating final samples...\n"
    print(final_sample_msg)
    stdout_file.write(final_sample_msg)
    stdout_file.flush()

    x, _ = next(it)
    x = jnp.asarray(x)

    with jax.disable_jit():
        recon = model.apply({"params": state.params}, x[:16])
        recon_np = np.array(recon).reshape(16, 28, 28)
        orig_np = np.array(x[:16]).reshape(16, 28, 28)

        plot_samples(
            np.concatenate([orig_np, recon_np], axis=0),
            np.concatenate([np.zeros(16, dtype=np.int32), np.ones(16, dtype=np.int32)]),
            samples_dir / f"step_{n_steps:04d}_final.png",
        )

    save_checkpoint(checkpoints_dir / f"step_{n_steps:05d}.msgpack", state)

    logger.close()

    final_msg = (
        "=" * 60 + "\n"
        "TRAINING COMPLETED!\n"
        f"Final average loss: {loss_avg:.6f}\n"
        f"Samples saved to: {samples_dir}\n"
        f"Checkpoints saved to: {checkpoints_dir}\n"
        f"Logs saved to: {logs_dir}\n"
        "=" * 60
    )
    print(final_msg)
    stdout_file.write(final_msg + "\n")
    stdout_file.flush()
    stdout_file.close()
    stderr_file.close()


if __name__ == "__main__":
    main()

