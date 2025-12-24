"""Super basic flow matching trainer from scratch."""

from typing import Optional

import jax
import jax.numpy as jnp
import optax

from meanflow_audio_codec.models import ConditionalFlow, TrainState
from meanflow_audio_codec.utils import sample_tr


def normalized_mse_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Normalized MSE loss as used in flow matching papers.
    
    Normalizes by the target's squared L2 norm to make the loss scale-invariant.
    Loss = ||pred - target||^2 / (||target||^2 + epsilon)
    
    Args:
        pred: Predicted values [B, ...]
        target: Target values [B, ...]
        epsilon: Small constant to prevent division by zero
    
    Returns:
        Normalized MSE loss (scalar)
    """
    delta = pred - target
    # Compute per-example squared L2 norm of error
    per_example_error = jnp.sum(delta ** 2, axis=tuple(range(1, delta.ndim)))
    
    # Compute per-example squared L2 norm of target
    per_example_target_norm = jnp.sum(target ** 2, axis=tuple(range(1, target.ndim)))
    
    # Normalize by target norm (with epsilon for numerical stability)
    normalized_per_example = per_example_error / (per_example_target_norm + epsilon)
    
    # Return mean over batch
    return jnp.mean(normalized_per_example)


def sample_time(key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
    """Sample time t uniformly from [0, 1]."""
    return jax.random.uniform(key, (batch_size, 1), dtype=jnp.float32)


def interpolate(x0: jnp.ndarray, x1: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation: (1-t) * x0 + t * x1."""
    return (1.0 - t) * x0 + t * x1


def compute_velocity_target(x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
    """Compute velocity target: v = x1 - x0."""
    return x1 - x0


@jax.jit
def train_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    x0: jnp.ndarray,
) -> tuple[TrainState, jnp.ndarray, jax.random.PRNGKey]:
    """Basic flow matching training step.
    
    Args:
        state: Training state with model and optimizer
        key: Random key
        x0: Data samples [B, D]
    
    Returns:
        Updated state, loss value, new random key
    """
    batch_size = x0.shape[0]
    
    # Split keys
    key, k_time, k_noise = jax.random.split(key, 3)
    
    # Sample time t ~ Uniform[0, 1]
    t = sample_time(k_time, batch_size)
    
    # Sample noise x1 ~ N(0, I)
    x1 = jax.random.normal(k_noise, x0.shape, dtype=x0.dtype)
    
    # Interpolate: x_t = (1-t) * x0 + t * x1
    x_t = interpolate(x0, x1, t)
    
    # Target velocity: v = x1 - x0
    v_target = compute_velocity_target(x0, x1)
    
    # Prepare time for model (needs [B, 2] with t and h)
    # For basic flow matching, h = 0
    time = jnp.concatenate([t, jnp.zeros_like(t)], axis=-1)
    
    def loss_fn(params):
        # Encode clean data to get latent
        latents = state.apply_fn(
            {"params": params}, x0, method="encode"
        )
        
        # Predict velocity at x_t, time t
        v_pred = state.apply_fn({"params": params}, x_t, time, latents)
        
        # Normalized MSE loss (as stated in the paper)
        return normalized_mse_loss(v_pred, v_target)
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update state
    state = state.apply_gradients(grads=grads)
    
    return state, loss, key


@jax.jit
def train_step_improved_mean_flow(
    state: TrainState,
    key: jax.random.PRNGKey,
    x0: jnp.ndarray,
) -> tuple[TrainState, jnp.ndarray, jax.random.PRNGKey]:
    """Improved Mean Flow training step.
    
    Uses the improved mean flow objective which samples both time t and
    reference time r, then uses JVP to compute the derivative for better
    gradient flow.
    
    Args:
        state: Training state with model and optimizer
        key: Random key
        x0: Data samples [B, D]
    
    Returns:
        Updated state, loss value, new random key
    """
    batch_size = x0.shape[0]
    
    # Split keys
    key, k_noise, k_tr = jax.random.split(key, 3)
    
    # Sample noise x1 ~ N(0, I)
    x1 = jax.random.normal(k_noise, x0.shape, dtype=x0.dtype)
    
    # Sample time t and reference time r using sample_tr
    time, r = sample_tr(k_tr, batch_size, dtype=x0.dtype)
    
    # Interpolate: x_t = (1-t) * x0 + t * x1
    # Using the same interpolation as baseline but with time from sample_tr
    x_t = (1.0 - time) * x0 + (0.001 + 0.999 * time) * x1
    
    # Target velocity: v = 0.999 * x1 - x0
    v_target = 0.999 * x1 - x0
    
    def loss_fn(params):
        # Encode clean data to get latent
        latents = state.apply_fn(
            {"params": params}, x0, method="encode"
        )
        
        # Define u function for JVP
        def u_fn(z_local, t_local, r_local):
            h_local = t_local - r_local
            th = jnp.concatenate([t_local, h_local], axis=-1)
            return state.apply_fn({"params": params}, z_local, th, latents)
        
        # Get initial velocity prediction at (x_t, t, 0)
        t_pair = jnp.concatenate([time, jnp.zeros_like(time)], axis=-1)
        v = state.apply_fn({"params": params}, x_t, t_pair, latents)
        
        # Compute JVP: (u, dudt) = JVP of u_fn at (x_t, time, r) in direction (v, 1, 0)
        (u, dudt) = jax.jvp(
            u_fn,
            (x_t, time, r),
            (v, jnp.ones_like(time), jnp.zeros_like(r)),
        )
        
        # Improved mean flow prediction: v_pred = u + (t - r) * stop_gradient(dudt)
        v_pred = u + (time - r) * jax.lax.stop_gradient(dudt)
        
        # Normalized MSE loss (as stated in the paper)
        return normalized_mse_loss(v_pred, v_target)
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update state
    state = state.apply_gradients(grads=grads)
    
    return state, loss, key


def create_train_state(
    model: ConditionalFlow,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    key: Optional[jax.random.PRNGKey] = None,
    batch_size: int = 32,
) -> TrainState:
    """Create initial training state.
    
    Args:
        model: ConditionalFlow model
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        key: Random key for initialization
        batch_size: Batch size for dummy initialization
    
    Returns:
        Initial TrainState
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Create optimizer
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    # Initialize model parameters
    key, k1, k2 = jax.random.split(key, 3)
    
    # Dummy inputs for initialization
    dummy_x = jnp.zeros((batch_size, model.noise_dimension), dtype=jnp.float32)
    dummy_t = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    dummy_latents = jnp.zeros((batch_size, model.latent_dimension), dtype=jnp.float32)
    
    # Initialize encoder
    params_enc = model.init(k1, dummy_x, method="encode")["params"]
    
    # Initialize decoder
    params_dec = model.init(k2, dummy_x, dummy_t, dummy_latents)["params"]
    
    # Merge parameters
    params = {**params_dec, "encoder": params_enc["encoder"]}
    
    # Create state
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    return state


def train_epoch(
    state: TrainState,
    data_iterator,
    key: jax.random.PRNGKey,
    n_steps: Optional[int] = None,
    use_improved_mean_flow: bool = False,
) -> tuple[TrainState, float, jax.random.PRNGKey]:
    """Train for one epoch.
    
    Args:
        state: Training state
        data_iterator: Iterator yielding batches of data
        key: Random key
        n_steps: Number of steps (None = use all data)
        use_improved_mean_flow: If True, use improved mean flow objective
    
    Returns:
        Updated state, average loss, new random key
    """
    total_loss = 0.0
    n_batches = 0
    
    # Select training step function
    step_fn = train_step_improved_mean_flow if use_improved_mean_flow else train_step
    
    for step, batch in enumerate(data_iterator):
        if n_steps is not None and step >= n_steps:
            break
        
        # Assume batch is (images, labels) or just images
        if isinstance(batch, tuple):
            x0 = batch[0]
        else:
            x0 = batch
        
        # Ensure x0 is JAX array
        if not isinstance(x0, jnp.ndarray):
            x0 = jnp.asarray(x0)
        
        # Flatten if needed (e.g., images to vectors)
        if x0.ndim > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        
        # Training step
        state, loss, key = step_fn(state, key, x0)
        
        total_loss += float(loss)
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return state, avg_loss, key

