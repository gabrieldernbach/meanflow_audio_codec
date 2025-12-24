#!/usr/bin/env python3
"""Minimal trial run of ConditionalConvFlow on MNIST.

Project-specific training script moved from root level.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds

from meanflow_audio_codec.datasets.mnist import preprocess_images
from meanflow_audio_codec.evaluators.sampling import sample
from meanflow_audio_codec.models import ConditionalConvFlow, TrainState
from meanflow_audio_codec.trainers.training_steps import train_step_improved_mean_flow
from meanflow_audio_codec.utils import ema


def load_data_simple(data_dir: str, batch_size: int, num_samples: int):
    """Load MNIST data in a simple, reliable way."""
    print(f"Loading {num_samples} samples from MNIST...")
    builder = tfds.builder("mnist", data_dir=data_dir)
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", as_supervised=True, shuffle_files=True)

    # Pre-load data into memory
    data_list = []
    for i, (img, label) in enumerate(tfds.as_numpy(ds.shuffle(60_000))):
        if i >= num_samples:
            break
        data_list.append((img, label))
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{num_samples} samples...")

    print(f"Loaded {len(data_list)} samples. Creating batches...")

    # Create batch iterator
    def batch_iterator():
        idx = 0
        while True:
            batch_imgs = []
            batch_labels = []
            for _ in range(batch_size):
                img, label = data_list[idx % len(data_list)]
                batch_imgs.append(img)
                batch_labels.append(label)
                idx += 1
            yield np.stack(batch_imgs), np.stack(batch_labels)

    return batch_iterator()


def main():
    """Main entry point for MNIST trial."""
    # Extended training configuration
    batch_size = 32  # Smaller batch for M1 Mac
    n_steps = 1000  # More steps to allow learning
    sample_every = 100
    sample_steps = 50  # More steps for better sample quality
    base_lr = 2e-4  # Slightly higher learning rate
    weight_decay = 1e-4
    seed = 42
    start_step = 0  # Can resume from checkpoint

    # Model hyperparameters - increased capacity
    noise_dimension = 784  # 28x28 for MNIST
    condition_dimension = 128  # Increased
    latent_dimension = 256  # Increased
    num_blocks = 4  # More blocks for capacity
    image_size = 28

    print("=" * 60)
    print("EXTENDED TRAINING: ConditionalConvFlow on MNIST")
    print("=" * 60)
    print(f"Steps: {n_steps}, Batch size: {batch_size}")
    print(
        f"Model: {num_blocks} blocks, "
        f"cond_dim={condition_dimension}, "
        f"latent_dim={latent_dimension}"
    )
    print(f"Learning rate: {base_lr}, Sample steps: {sample_steps}")
    print("-" * 60)

    # Create model
    print("Initializing model...")
    model = ConditionalConvFlow(
        noise_dimension=noise_dimension,
        condition_dimension=condition_dimension,
        latent_dimension=latent_dimension,
        num_blocks=num_blocks,
        image_size=image_size,
        use_grn=True,
    )

    # Setup optimizer
    tx = optax.adamw(learning_rate=base_lr, weight_decay=weight_decay)

    # Initialize model
    print("Initializing parameters...")
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)

    x0 = jnp.zeros((batch_size, model.noise_dimension), dtype=jnp.float32)
    t0 = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    params = model.init(k_init, x0, t0)["params"]
    print("✓ Model initialized")

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Create output directory (using workdir pattern)
    workdir = Path("./outputs/trial_conv_flow")
    samples_dir = workdir / "samples"
    checkpoints_dir = workdir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint loading would go here if needed
    start_step = 0

    # Load data
    data_dir = str(Path.home() / "datasets" / "mnist")
    num_data_samples = max(10000, n_steps * batch_size)  # Enough for training
    it = load_data_simple(data_dir, batch_size, num_data_samples)

    print(f"Workdir: {workdir}")
    print("-" * 60)

    # Training loop
    loss_avg = None
    sample_key = jax.random.PRNGKey(seed + 1)

    print(f"\nStarting training from step {start_step}...")
    print(f"{'Step':<6} {'Loss':<12} {'Avg Loss':<12} {'Status'}")
    print("-" * 60)

    for step in range(start_step, n_steps):
        img, tar = next(it)

        x = preprocess_images(img, normalize=True)
        x = jnp.asarray(x)

        state, loss, key = train_step_improved_mean_flow(state, key, x)

        loss_val = float(loss)
        loss_avg = ema(loss_avg, loss_val)

        # Print progress periodically
        if step % 10 == 0:
            print(f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f}")

        # Sample and save checkpoint periodically
        if step % sample_every == 0 and step > 0:
            print(
                f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f} sampling..."
            )

            # TODO: Replace with actual encoder latents
            # For now, use zero latents as placeholder
            dummy_latents = jnp.zeros(
                (16, 32, latent_dimension),
                dtype=jnp.float32
            )
            smps = sample(
                state.apply_fn,
                model.noise_dimension,
                state.params,
                sample_key,
                latents=dummy_latents,
                n_steps=sample_steps,
                use_improved_mean_flow=True,
                guidance_scale=1.0,
            )
            n_show = min(16, len(smps))
            smps_np = np.array(smps[:n_show]).reshape(n_show, 28, 28)
            # Use dummy labels for plotting (no class conditioning)
            cls_np = np.zeros(n_show, dtype=np.int32)

            n_rows = int(np.ceil(np.sqrt(n_show)))
            n_cols = int(np.ceil(n_show / n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 6))
            axs = axs.flatten() if n_show > 1 else [axs]
            for ax, xt, lab in zip(axs[:n_show], smps_np, cls_np):
                ax.imshow(xt, vmin=-1, vmax=1, cmap="gray")
                ax.axis("off")
                ax.set_title(int(lab), fontsize=8)
            fig.tight_layout()
            fig.savefig(samples_dir / f"step_{step:04d}.png", dpi=100)
            plt.close(fig)

            # Save checkpoint
            from flax import serialization

            checkpoint_path = checkpoints_dir / "latest.msgpack"
            with checkpoint_path.open("wb") as f:
                f.write(serialization.to_bytes(state))
            print(
                f"{step:<6} {loss_val:<12.6f} {loss_avg:<12.6f} "
                f"✓ saved + checkpoint"
            )

    # Final sample
    print("-" * 60)
    print("Generating final samples...")
    # TODO: Replace with actual encoder latents
    # For now, use zero latents as placeholder
    dummy_latents = jnp.zeros(
        (16, 32, latent_dimension),
        dtype=jnp.float32
    )
    smps = sample(
        state.apply_fn,
        model.noise_dimension,
        state.params,
        sample_key,
        latents=dummy_latents,
        n_steps=sample_steps,
        use_improved_mean_flow=True,
        guidance_scale=1.0,
    )

    n_show = min(16, len(smps))
    smps_np = np.array(smps[:n_show]).reshape(n_show, 28, 28)
    # Use dummy labels for plotting (no class conditioning)
    cls_np = np.zeros(n_show, dtype=np.int32)

    n_rows = int(np.ceil(np.sqrt(n_show)))
    n_cols = int(np.ceil(n_show / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 6))
    axs = axs.flatten() if n_show > 1 else [axs]
    for ax, xt, lab in zip(axs[:n_show], smps_np, cls_np):
        ax.imshow(xt, vmin=-1, vmax=1, cmap="gray")
        ax.axis("off")
        ax.set_title(int(lab), fontsize=8)
    fig.tight_layout()
    fig.savefig(samples_dir / f"step_{n_steps:04d}_final.png", dpi=100)
    plt.close(fig)

    print("=" * 60)
    print("TRIAL COMPLETED!")
    print(f"Final average loss: {loss_avg:.6f}")
    print(f"Samples saved to: {samples_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

