"""Flow model training loop."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.datasets.mnist import load_mnist, preprocess_images
from meanflow_audio_codec.evaluators.sampling import sample
from meanflow_audio_codec.models import ConditionalFlow, TrainState
from meanflow_audio_codec.trainers.training_steps import (
    train_step, train_step_improved_mean_flow)
from meanflow_audio_codec.trainers.utils import (LogWriter, find_latest_checkpoint,
                                          load_checkpoint_and_resume,
                                          plot_samples, save_checkpoint)
from meanflow_audio_codec.utils import ema


def train_flow(config: TrainFlowConfig, resume: bool = False) -> None:
    """Train a flow model.

    Args:
        config: Training configuration
        resume: If True, attempt to resume from checkpoint in workdir
    """
    if config.condition_dimension % 2:
        raise ValueError(
            f"condition_dimension must be even, got {config.condition_dimension}"
        )

    # Validate data_dir - use default if not provided
    if config.data_dir is None:
        data_dir = str(Path.home() / "datasets" / "mnist")
    else:
        data_dir = config.data_dir

    # Setup workdir structure
    workdir = config.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    samples_dir = workdir / "samples"
    checkpoints_dir = workdir / "checkpoints"
    logs_dir = workdir / "logs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = ConditionalFlow(
        noise_dimension=config.noise_dimension,
        condition_dimension=config.condition_dimension,
        latent_dimension=config.latent_dimension,
        num_blocks=config.num_blocks,
    )

    tx = optax.adamw(learning_rate=config.base_lr, weight_decay=config.weight_decay)

    key = jax.random.PRNGKey(config.seed)
    key, k_init = jax.random.split(key)

    x0 = jnp.zeros((config.batch_size, model.noise_dimension), dtype=jnp.float32)
    t0 = jnp.zeros((config.batch_size, 2), dtype=jnp.float32)
    # Initialize both encoder and decoder paths
    # Encoder parameters are initialized when encode is called
    # Decoder parameters are initialized when __call__ is called
    # We need to initialize both to get all parameters
    key1, key2 = jax.random.split(k_init)
    # Initialize encoder
    params_enc = model.init(key1, x0, method="encode")["params"]
    
    # Initialize decoder with concatenated input to ensure correct block initialization
    # CRITICAL: Blocks must see concatenated input [B, latent_dim + noise_dim] during init
    # to ensure MLP layers get correct input_dimension
    # Using setup() in MLP instead of @nn.compact should help avoid shape inference issues
    dummy_latents = jnp.zeros((config.batch_size, config.latent_dimension), dtype=jnp.float32)
    # Initialize decoder using __call__ with latents - this ensures concatenation happens
    # and blocks are initialized with correct input_dimension
    params_dec = model.init(key2, x0, t0, dummy_latents)["params"]
    
    # Merge: encoder is under 'encoder' key, decoder has 'blocks' key
    # Since they share the encoder submodule, we use encoder params from encode init
    params = {**params_dec, "encoder": params_enc["encoder"]}

    state_template = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Handle checkpoint resume
    start_step = 0
    if resume:
        checkpoint_path = find_latest_checkpoint(workdir)
        if checkpoint_path is not None:
            state_template, start_step = load_checkpoint_and_resume(
                workdir, state_template
            )
            print(f"Resuming from checkpoint at step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    state = state_template

    # Setup data loading
    it = load_mnist(data_dir=data_dir, split="train", batch_size=config.batch_size)

    # Setup logging
    log_path = logs_dir / "train_log.jsonl"
    logger = LogWriter(log_path)

    loss_avg = None
    sample_key = jax.random.PRNGKey(config.sample_seed)

    checkpoint_step = (
        config.checkpoint_step if config.checkpoint_step is not None else config.n_steps
    )
    saved_checkpoint = False
    
    # Initialize profiling
    from meanflow_audio_codec.trainers.profiling import ProfilingTrainer
    profiler = ProfilingTrainer(logger)
    profiler.start_training(state.params)

    for step in range(start_step, config.n_steps):
        profiler.before_step(step)
        
        img, tar = next(it)

        x = preprocess_images(img, format="1d", normalize=True)
        x = jnp.asarray(x)

        if config.use_improved_mean_flow:
            state, loss, key = train_step_improved_mean_flow(state, key, x)
        else:
            state, loss, key = train_step(state, key, x)

        loss_val = float(loss)
        loss_avg = ema(loss_avg, loss_val)

        # Log to JSON with profiling
        profiler.after_step(
            step,
            {
                "loss": loss_val,
                "loss_avg": loss_avg if loss_avg is not None else loss_val,
                "lr": config.base_lr,
            },
        )

        # Print to stdout
        if step % 50 == 0:
            print(f"step={step:04d} loss={loss_val:.9f} loss_avg={loss_avg:.9f}")

        if step % config.sample_every == 0:
            # TODO: Replace with actual encoder latents (encode a reference image)
            # For now, use zero latents as placeholder
            dummy_latents = jnp.zeros(
                (config.batch_size, config.latent_dimension),
                dtype=jnp.float32
            )
            smps = sample(
                state.apply_fn,
                model.noise_dimension,
                state.params,
                sample_key,
                latents=dummy_latents,
                n_steps=config.sample_steps,
                use_improved_mean_flow=config.use_improved_mean_flow,
                guidance_scale=1.0,
            )
            n_show = min(16, len(smps))
            smps_np = np.array(smps[:n_show]).reshape(n_show, 28, 28)
            # Use dummy labels for plotting (no class conditioning)
            cls_np = np.zeros(n_show, dtype=np.int32)

            plot_samples(
                smps_np,
                cls_np,
                samples_dir / f"step_{step:04d}.png",
            )

        if step + 1 == checkpoint_step:
            save_checkpoint(checkpoints_dir / f"step_{step + 1:05d}.msgpack", state)
            saved_checkpoint = True

    # Final sample
    # TODO: Replace with actual encoder latents (encode a reference image)
    # For now, use zero latents as placeholder
    dummy_latents = jnp.zeros(
        (config.batch_size, config.latent_dimension),
        dtype=jnp.float32
    )
    smps = sample(
        state.apply_fn,
        model.noise_dimension,
        state.params,
        sample_key,
        latents=dummy_latents,
        n_steps=config.sample_steps,
        use_improved_mean_flow=config.use_improved_mean_flow,
        guidance_scale=1.0,
    )

    n_show = min(16, len(smps))
    smps_np = np.array(smps[:n_show]).reshape(n_show, 28, 28)
    # Use dummy labels for plotting (no class conditioning)
    cls_np = np.zeros(n_show, dtype=np.int32)

    plot_samples(
        smps_np,
        cls_np,
        samples_dir / f"step_{config.n_steps:04d}.png",
    )

    if not saved_checkpoint:
        save_checkpoint(checkpoints_dir / f"step_{config.n_steps:05d}.msgpack", state)
    
    # End profiling and log summary
    training_summary = profiler.end_training(config.n_steps)
    print(f"\nTraining Summary:")
    print(f"  Total time: {training_summary['total_training_time_hours']:.2f} hours")
    print(f"  Steps/sec: {training_summary['steps_per_second']:.2f}")
    print(f"  Avg step time: {training_summary['avg_step_time']*1000:.2f} ms")

    logger.close()

