"""Training step functions for flow models."""

import jax
import jax.numpy as jnp

from meanflow_audio_codec.models import TrainState
from meanflow_audio_codec.utils import logit_normal, sample_tr, weighted_l2_loss


@jax.jit
def train_step(state: TrainState, key: jax.random.PRNGKey, x: jnp.ndarray):
    """Baseline Flow Matching training step."""
    key, k_noise, k_time = jax.random.split(key, 3)
    noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
    time = logit_normal(k_time, (x.shape[0], 1), dtype=x.dtype)
    time = jnp.concatenate([time, jnp.zeros_like(time)], axis=-1)

    noised = (1.0 - time[:, :1]) * x + (0.001 + 0.999 * time[:, :1]) * noise
    target = 0.999 * noise - x  # matches: noise.mul(0.999).sub(x)

    def loss_fn(params):
        # Encode clean image to get latent
        latents = state.apply_fn(
            {"params": params}, x, method="encode"
        )
        # Decode from latent + noise
        pred = state.apply_fn({"params": params}, noised, time, latents)
        return weighted_l2_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, key


@jax.jit
def train_step_improved_mean_flow(
    state: TrainState,
    key: jax.random.PRNGKey,
    x: jnp.ndarray,
):
    """Improved Mean Flow training step."""
    key, k_noise, k_tr = jax.random.split(key, 3)
    noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
    time, r = sample_tr(k_tr, x.shape[0], dtype=x.dtype)

    noised = (1.0 - time) * x + (0.001 + 0.999 * time) * noise
    target = 0.999 * noise - x

    def loss_fn(params):
        # Encode clean image to get latent
        latents = state.apply_fn(
            {"params": params}, x, method="encode"
        )
        
        def u_fn(z_local, t_local, r_local):
            h_local = t_local - r_local
            th = jnp.concatenate([t_local, h_local], axis=-1)
            return state.apply_fn({"params": params}, z_local, th, latents)

        t_pair = jnp.concatenate([time, jnp.zeros_like(time)], axis=-1)
        v = state.apply_fn({"params": params}, noised, t_pair, latents)

        (u, dudt) = jax.jvp(
            u_fn,
            (noised, time, r),
            (v, jnp.ones_like(time), jnp.zeros_like(r)),
        )
        v_pred = u + (time - r) * jax.lax.stop_gradient(dudt)
        return weighted_l2_loss(v_pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, key

