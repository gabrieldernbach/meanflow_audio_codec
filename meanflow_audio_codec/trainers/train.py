"""Flow model training loop."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.datasets.audio import build_audio_pipeline
from meanflow_audio_codec.datasets.mnist import load_mnist
from meanflow_audio_codec.evaluators.sampling import sample
from meanflow_audio_codec.models import ConditionalFlow, TrainState
from meanflow_audio_codec.preprocessing.tokenization_utils import (
    compute_token_shape,
    compute_tokenized_dimension,
    create_tokenization_strategy,
)
from meanflow_audio_codec.trainers.loss_strategies import (
    FlowMatchingLoss,
    ImprovedMeanFlowLoss,
    LossStrategy,
    MeanFlowLoss,
)
from meanflow_audio_codec.trainers.noise_schedules import (
    LinearNoiseSchedule,
    UniformNoiseSchedule,
)
from meanflow_audio_codec.trainers.time_sampling import (
    LogitNormalTimeSampling,
    MeanFlowTimeSampling,
    UniformTimeSampling,
)
from meanflow_audio_codec.trainers.training_steps import (
    train_step, train_step_improved_mean_flow)
from meanflow_audio_codec.trainers.utils import (
    LogWriter,
    cleanup_old_checkpoints,
    collect_experiment_metadata,
    find_latest_checkpoint,
    generate_config_diff,
    generate_training_summary,
    load_checkpoint_and_resume,
    plot_samples,
    save_checkpoint_with_metadata,
    save_json,
)
from meanflow_audio_codec.utils import ema


def create_loss_strategy(config: TrainFlowConfig) -> LossStrategy:
    """Create loss strategy from configuration.
    
    Args:
        config: Training configuration
    
    Returns:
        Loss strategy instance
    """
    # Determine loss strategy type
    loss_strategy_name = config.loss_strategy
    if loss_strategy_name is None:
        # Infer from use_improved_mean_flow for backward compatibility
        loss_strategy_name = (
            "improved_mean_flow" if config.use_improved_mean_flow else "flow_matching"
        )
    
    # Create noise schedule
    noise_schedule_name = config.noise_schedule or "linear"
    if noise_schedule_name == "linear":
        noise_min = config.noise_min if config.noise_min is not None else 0.001
        noise_max = config.noise_max if config.noise_max is not None else 0.999
        noise_schedule = LinearNoiseSchedule(noise_min=noise_min, noise_max=noise_max)
    elif noise_schedule_name == "uniform":
        noise_schedule = UniformNoiseSchedule()
    else:
        raise ValueError(
            f"Unknown noise_schedule: {noise_schedule_name}. "
            "Must be one of: 'linear', 'uniform'"
        )
    
    # Create time sampling strategy
    time_sampling_name = config.time_sampling or "logit_normal"
    if time_sampling_name == "uniform":
        time_sampling = UniformTimeSampling()
    elif time_sampling_name == "logit_normal":
        mean = config.time_sampling_mean if config.time_sampling_mean is not None else -0.4
        std = config.time_sampling_std if config.time_sampling_std is not None else 1.0
        time_sampling = LogitNormalTimeSampling(mean=mean, std=std)
    elif time_sampling_name == "mean_flow":
        mean = config.time_sampling_mean if config.time_sampling_mean is not None else -0.4
        std = config.time_sampling_std if config.time_sampling_std is not None else 1.0
        data_proportion = (
            config.time_sampling_data_proportion
            if config.time_sampling_data_proportion is not None
            else 0.5
        )
        time_sampling = MeanFlowTimeSampling(
            mean=mean, std=std, data_proportion=data_proportion
        )
    else:
        raise ValueError(
            f"Unknown time_sampling: {time_sampling_name}. "
            "Must be one of: 'uniform', 'logit_normal', 'mean_flow'"
        )
    
    # Determine if weighted loss should be used
    use_weighted_loss = (
        config.use_weighted_loss if config.use_weighted_loss is not None else True
    )
    
    # Create loss strategy
    if loss_strategy_name == "flow_matching":
        return FlowMatchingLoss(
            noise_schedule=noise_schedule,
            time_sampling=time_sampling,
            use_weighted_loss=use_weighted_loss,
        )
    elif loss_strategy_name == "mean_flow":
        # For mean flow, time_sampling must be MeanFlowTimeSampling
        if not isinstance(time_sampling, MeanFlowTimeSampling):
            time_sampling = MeanFlowTimeSampling(
                mean=config.time_sampling_mean or -0.4,
                std=config.time_sampling_std or 1.0,
                data_proportion=config.time_sampling_data_proportion or 0.5,
            )
        gamma = config.gamma if config.gamma is not None else 0.5
        c = config.c if config.c is not None else 1e-3
        return MeanFlowLoss(
            noise_schedule=noise_schedule,
            time_sampling=time_sampling,
            gamma=gamma,
            c=c,
        )
    elif loss_strategy_name == "improved_mean_flow":
        # For improved mean flow, time_sampling must be MeanFlowTimeSampling
        if not isinstance(time_sampling, MeanFlowTimeSampling):
            time_sampling = MeanFlowTimeSampling(
                mean=config.time_sampling_mean or -0.4,
                std=config.time_sampling_std or 1.0,
                data_proportion=config.time_sampling_data_proportion or 0.5,
            )
        return ImprovedMeanFlowLoss(
            noise_schedule=noise_schedule,
            time_sampling=time_sampling,
            use_weighted_loss=use_weighted_loss,
        )
    else:
        raise ValueError(
            f"Unknown loss_strategy: {loss_strategy_name}. "
            "Must be one of: 'flow_matching', 'mean_flow', 'improved_mean_flow'"
        )


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

    # Validate data_dir is provided
    if config.data_dir is None:
        raise ValueError(
            "config.data_dir must be provided. It cannot be None. "
            "Please specify a valid data directory path in your configuration."
        )
    
    data_dir = config.data_dir

    # Setup tokenization strategy
    tokenization = create_tokenization_strategy(config)
    
    # Determine effective noise dimension (tokenized if tokenization is used)
    original_noise_dim = config.noise_dimension
    token_shape = None
    if tokenization is not None:
        dataset_name = config.dataset or "mnist"
        effective_noise_dim = compute_tokenized_dimension(
            tokenization, original_noise_dim, dataset_name
        )
        token_shape = compute_token_shape(tokenization, original_noise_dim, dataset_name)
        print(f"Using tokenization: {config.tokenization_strategy}")
        print(f"Original dimension: {original_noise_dim} -> Tokenized dimension: {effective_noise_dim}")
        print(f"Token shape: {token_shape} (n_tokens, token_dim)")
    else:
        effective_noise_dim = original_noise_dim
        print(f"No tokenization, using original dimension: {original_noise_dim}")

    # Setup workdir structure
    workdir = config.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    samples_dir = workdir / "samples"
    checkpoints_dir = workdir / "checkpoints"
    logs_dir = workdir / "logs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Collect and save experiment metadata
    metadata = collect_experiment_metadata(config)
    save_json(workdir / "metadata.json", metadata.to_dict())

    # Save config
    save_json(workdir / "config.json", config.to_dict())

    # Generate config diff if resuming
    if resume:
        config_diff = generate_config_diff(workdir, config)
        if config_diff:
            save_json(workdir / "config_diff.json", config_diff)
            print("Config differences from previous run:")
            if config_diff.get("changed"):
                print("  Changed parameters:")
                for key, change in config_diff["changed"].items():
                    print(f"    {key}: {change['old']} -> {change['new']}")
            if config_diff.get("added"):
                print(f"  Added parameters: {', '.join(config_diff['added'])}")
            if config_diff.get("removed"):
                print(f"  Removed parameters: {', '.join(config_diff['removed'])}")

    # Initialize model with effective noise dimension (tokenized if applicable)
    model = ConditionalFlow(
        noise_dimension=effective_noise_dim,
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
        try:
            state_template, start_step = load_checkpoint_and_resume(
                workdir, state_template, config
            )
            print(f"Resuming from checkpoint at step {start_step}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to resume from checkpoint: {e}")
            print("Starting from scratch")
            start_step = 0

    state = state_template

    # Setup data loading based on dataset
    dataset_name = config.dataset or "mnist"
    if dataset_name == "mnist":
        it = load_mnist(data_dir=data_dir, split="train", batch_size=config.batch_size)
        # Extract images from iterator (MNIST returns (images, labels))
        def data_iterator():
            for img, _ in it:
                yield img
        it = data_iterator()
    elif dataset_name == "audio":
        # Audio pipeline returns batches directly
        it = build_audio_pipeline(
            data_dir=data_dir,
            seed=config.seed,
            frame_sz=config.noise_dimension,  # Use original noise_dimension for frame size
            batch_size=config.batch_size,
        )
        # Flatten audio frames to [B, frame_sz]
        def data_iterator():
            for batch in it:
                # batch is [B, frame_sz, n_channels], flatten to [B, frame_sz * n_channels]
                # or keep as [B, frame_sz] if mono
                if batch.ndim == 3:
                    batch = batch.reshape(batch.shape[0], -1)
                yield batch
        it = data_iterator()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Setup logging
    log_path = logs_dir / "train_log.jsonl"
    logger = LogWriter(log_path)

    loss_avg = None
    sample_key = jax.random.PRNGKey(config.sample_seed)

    checkpoint_step = (
        config.checkpoint_step if config.checkpoint_step is not None else config.n_steps
    )
    saved_checkpoint = False
    
    # Create loss strategy from config (create once, reuse in loop)
    loss_strategy = create_loss_strategy(config)

    # Initialize profiling
    from meanflow_audio_codec.trainers.profiling import ProfilingTrainer
    profiler = ProfilingTrainer(logger)
    profiler.start_training(state.params)

    for step in range(start_step, config.n_steps):
        profiler.before_step(step)
        
        batch_data = next(it)
        x = jnp.asarray(batch_data)  # Convert to JAX array
        
        # Apply tokenization if configured
        if tokenization is not None:
            # Tokenize: [B, original_dim] -> [B, n_tokens, token_dim]
            x_tokens = tokenization.tokenize(x)
            # Flatten: [B, n_tokens, token_dim] -> [B, n_tokens * token_dim]
            x = x_tokens.reshape(x_tokens.shape[0], -1)

        # Use unified training step with loss strategy
        # Strategy is created once outside loop for JIT efficiency
        state, loss, key = train_step(state, key, x, loss_strategy)

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
            
            # Handle detokenization if tokenization is used
            if tokenization is not None and token_shape is not None:
                # Unflatten: [B, n_tokens * token_dim] -> [B, n_tokens, token_dim]
                smps_tokens = smps.reshape(smps.shape[0], token_shape[0], token_shape[1])
                # Detokenize: [B, n_tokens, token_dim] -> [B, original_dim]
                smps = tokenization.detokenize(smps_tokens)
            
            n_show = min(16, len(smps))
            # Reshape for plotting (assume square images for MNIST)
            if config.dataset == "mnist" or config.dataset is None:
                img_size = int(original_noise_dim ** 0.5)
                smps_np = np.array(smps[:n_show]).reshape(n_show, img_size, img_size)
            else:
                # For audio or other datasets, just use flattened
                smps_np = np.array(smps[:n_show])
            # Use dummy labels for plotting (no class conditioning)
            cls_np = np.zeros(n_show, dtype=np.int32)

            plot_samples(
                smps_np,
                cls_np,
                samples_dir / f"step_{step:04d}.png",
            )

        if step + 1 == checkpoint_step:
            checkpoint_path = checkpoints_dir / f"step_{step + 1:05d}.msgpack"
            save_checkpoint_with_metadata(checkpoint_path, state, step + 1, config)
            saved_checkpoint = True
            
            # Cleanup old checkpoints if configured
            if config.max_checkpoints_to_keep is not None:
                removed = cleanup_old_checkpoints(
                    workdir,
                    config.max_checkpoints_to_keep,
                    keep_final=False,  # Don't keep final yet, we're still training
                    final_step=None,
                )
                if removed:
                    print(f"Cleaned up {len(removed)} old checkpoint(s)")

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
    
    # Handle detokenization if tokenization is used
    if tokenization is not None and token_shape is not None:
        # Unflatten: [B, n_tokens * token_dim] -> [B, n_tokens, token_dim]
        smps_tokens = smps.reshape(smps.shape[0], token_shape[0], token_shape[1])
        # Detokenize: [B, n_tokens, token_dim] -> [B, original_dim]
        smps = tokenization.detokenize(smps_tokens)
    
    n_show = min(16, len(smps))
    # Reshape for plotting (assume square images for MNIST)
    if config.dataset == "mnist" or config.dataset is None:
        img_size = int(original_noise_dim ** 0.5)
        smps_np = np.array(smps[:n_show]).reshape(n_show, img_size, img_size)
    else:
        # For audio or other datasets, just use flattened
        smps_np = np.array(smps[:n_show])
    # Use dummy labels for plotting (no class conditioning)
    cls_np = np.zeros(n_show, dtype=np.int32)

    plot_samples(
        smps_np,
        cls_np,
        samples_dir / f"step_{config.n_steps:04d}.png",
    )

    if not saved_checkpoint:
        checkpoint_path = checkpoints_dir / f"step_{config.n_steps:05d}.msgpack"
        save_checkpoint_with_metadata(checkpoint_path, state, config.n_steps, config)
    
    # Final cleanup if configured
    if config.max_checkpoints_to_keep is not None:
        removed = cleanup_old_checkpoints(
            workdir,
            config.max_checkpoints_to_keep,
            keep_final=True,
            final_step=config.n_steps,
        )
        if removed:
            print(f"Cleaned up {len(removed)} old checkpoint(s)")
    
    # End profiling and log summary
    training_summary = profiler.end_training(config.n_steps)
    
    # Generate and save training summary from logs
    log_path = logs_dir / "train_log.jsonl"
    metrics_summary = generate_training_summary(log_path)
    
    # Merge profiling summary with metrics summary
    full_summary = {
        "profiling": training_summary,
        "metrics": metrics_summary,
    }
    save_json(workdir / "summary.json", full_summary)
    
    print(f"\nTraining Summary:")
    print(f"  Total time: {training_summary['total_training_time_hours']:.2f} hours")
    print(f"  Steps/sec: {training_summary['steps_per_second']:.2f}")
    print(f"  Avg step time: {training_summary['avg_step_time']*1000:.2f} ms")
    
    if "best_loss" in metrics_summary:
        print(f"  Best loss: {metrics_summary['best_loss']['value']:.6f} at step {metrics_summary['best_loss']['step']}")
    if "convergence" in metrics_summary:
        conv = metrics_summary["convergence"]
        print(f"  Loss improvement: {conv['improvement']:.6f} ({conv['improvement_percent']:.2f}%)")
    if "loss_statistics" in metrics_summary:
        stats = metrics_summary["loss_statistics"]
        print(f"  Loss stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}")

    logger.close()

