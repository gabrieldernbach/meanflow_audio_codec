#!/usr/bin/env python3
"""Train MLPMixerAutoencoder on audio with L2 reconstruction loss.

Project-specific training script for audio autoencoder.
Method (A) Pure Autoencoder on Audio dataset.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from meanflow_audio_codec.datasets import build_audio_pipeline
from meanflow_audio_codec.models import MLPMixerAutoencoder, TrainState
from meanflow_audio_codec.trainers.utils import (
    LogWriter,
    find_latest_checkpoint,
    load_checkpoint_and_resume,
    save_checkpoint,
)
from meanflow_audio_codec.utils import ema


@jax.jit
def train_step_autoencoder(state: TrainState, x: jnp.ndarray) -> tuple[TrainState, jnp.ndarray]:
    """Autoencoder training step with L2 reconstruction loss."""
    def loss_fn(params):
        recon = state.apply_fn({"params": params}, x)
        loss = jnp.mean((recon - x) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    # Training configuration - appropriate for M1 MacBook 16GB RAM
    batch_size = 8  # Smaller batch for audio frames
    n_steps = 2000
    sample_every = 200
    base_lr = 1e-3
    weight_decay = 1e-4
    seed = 42
    resume = False

    # Audio frame configuration
    frame_sz = 16384  # Samples per frame (smaller for development)
    n_channels = 2  # Stereo
    input_dim = frame_sz * n_channels  # 32768

    # Model hyperparameters - adjusted for audio input dimension
    num_latent_tokens = 16
    latent_dim = 64
    num_context_tokens = 128
    num_output_tokens = 128
    token_mix_dim = 256
    channel_mix_dim = 256

    # Data directory - global convention
    data_dir = Path.home() / "datasets" / "wavegen"

    # Setup workdir
    workdir = Path("./outputs/trial_mlp_mixer_autoencoder_audio")
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
        "TRAINING: MLPMixerAutoencoder on Audio\n"
        "=" * 60 + "\n"
        f"Steps: {n_steps}, Batch size: {batch_size}\n"
        f"Audio: frame_sz={frame_sz}, n_channels={n_channels}, input_dim={input_dim}\n"
        f"Model: num_latent_tokens={num_latent_tokens}, latent_dim={latent_dim}\n"
        f"num_context_tokens={num_context_tokens}, num_output_tokens={num_output_tokens}\n"
        f"token_mix_dim={token_mix_dim}, channel_mix_dim={channel_mix_dim}\n"
        f"Learning rate: {base_lr}\n"
        f"Data directory: {data_dir}\n"
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
                state_template, start_step = load_checkpoint_and_resume(workdir, state_template)
                print(f"✓ Resuming from checkpoint at step {start_step}")
            except Exception as e:
                print(f"⚠ Failed to load checkpoint: {e}")
                print("Starting from scratch instead...")
                start_step = 0
        else:
            print("No valid checkpoint found, starting from scratch")

    state = state_template

    # Load audio data
    print(f"Loading audio dataset from {data_dir}...")
    audio_pipeline = build_audio_pipeline(
        data_dir=str(data_dir),
        seed=seed,
        frame_sz=frame_sz,
        prefetch=4,
        buffer_size=1000,
        batch_size=batch_size,
        drop_last=True,
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
        # Get audio batch: shape (batch_size, frame_sz, n_channels)
        audio_batch = next(audio_pipeline)
        audio_batch = jnp.asarray(audio_batch, dtype=jnp.float32)

        # Flatten to (batch_size, frame_sz * n_channels) for autoencoder
        x = audio_batch.reshape(batch_size, -1)

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
                recon = model.apply({"params": state.params}, x[:4])
                recon_np = np.array(recon).reshape(4, frame_sz, n_channels)
                orig_np = np.array(x[:4]).reshape(4, frame_sz, n_channels)

                # Save audio samples (first channel only for visualization)
                # Note: For actual audio playback, you'd want to save the full stereo signal
                sample_data = {
                    "step": step,
                    "original": orig_np[:, :, 0].tolist(),  # First channel
                    "reconstructed": recon_np[:, :, 0].tolist(),  # First channel
                }
                sample_path = samples_dir / f"step_{step:04d}.json"
                with open(sample_path, "w") as f:
                    json.dump(sample_data, f)

            # Save checkpoint
            save_checkpoint(checkpoints_dir / f"step_{step:05d}.msgpack", state)
            save_checkpoint(checkpoints_dir / "latest.msgpack", state)
            msg = f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f} ✓ saved"
            print(msg)
            stdout_file.write(msg + "\n")
            stdout_file.flush()

    # Final sample
    final_sample_msg = "-" * 60 + "\nGenerating final samples...\n"
    print(final_sample_msg)
    stdout_file.write(final_sample_msg)
    stdout_file.flush()

    audio_batch = next(audio_pipeline)
    audio_batch = jnp.asarray(audio_batch, dtype=jnp.float32)
    x = audio_batch.reshape(batch_size, -1)

    with jax.disable_jit():
        recon = model.apply({"params": state.params}, x[:4])
        recon_np = np.array(recon).reshape(4, frame_sz, n_channels)
        orig_np = np.array(x[:4]).reshape(4, frame_sz, n_channels)

        sample_data = {
            "step": n_steps,
            "original": orig_np[:, :, 0].tolist(),
            "reconstructed": recon_np[:, :, 0].tolist(),
        }
        sample_path = samples_dir / f"step_{n_steps:04d}_final.json"
        with open(sample_path, "w") as f:
            json.dump(sample_data, f)

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

