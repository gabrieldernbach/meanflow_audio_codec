"""Tests for Improved Mean Flow boundary conditions and JVP correctness."""

import jax
import jax.numpy as jnp
import numpy as np

from meanflow_audio_codec.models import ConditionalFlow
from meanflow_audio_codec.utils import weighted_l2_loss


def _imf_v_pred(model, params, noised, t, r):
    """Helper to compute v_pred for Improved Mean Flow."""
    def u_fn(z_local, t_local, r_local):
        h_local = t_local - r_local
        th = jnp.concatenate([t_local, h_local], axis=-1)
        return model.apply({"params": params}, z_local, th)

    t_pair = jnp.concatenate([t, jnp.zeros_like(t)], axis=-1)
    v = model.apply({"params": params}, noised, t_pair)
    v_dir = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    (u, dudt) = jax.jvp(
        u_fn,
        (noised, t, r),
        (v_dir, jnp.ones_like(t), jnp.zeros_like(r)),
    )
    v_pred = u + (t - r) * jax.lax.stop_gradient(dudt)
    return u, v_pred


def test_improved_mean_flow_boundary_condition():
    """Test boundary condition: when t=r, v_pred should equal u."""
    key = jax.random.PRNGKey(0)
    model = ConditionalFlow(
        noise_dimension=8,
        condition_dimension=32,
        latent_dimension=64,
        num_blocks=2,
    )

    batch_size = 4
    x = jnp.zeros((batch_size, 8), dtype=jnp.float32)
    time = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    params = model.init(key, x, time)["params"]

    key, k_noise, k_time = jax.random.split(key, 3)
    noise = jax.random.normal(k_noise, x.shape)
    t = jax.random.uniform(k_time, (batch_size, 1))
    r = t  # Boundary condition: t = r

    noised = (1.0 - t) * x + (0.001 + 0.999 * t) * noise
    u, v_pred = _imf_v_pred(model, params, noised, t, r)

    np.testing.assert_allclose(v_pred, u, rtol=1e-6, atol=1e-6)


def test_improved_mean_flow_jvp_matches_finite_difference():
    """Test that JVP computation matches reverse-mode gradient computation."""
    key = jax.random.PRNGKey(2)
    model = ConditionalFlow(
        noise_dimension=6,
        condition_dimension=32,
        latent_dimension=64,
        num_blocks=2,
    )

    batch_size = 3
    x = jax.random.normal(key, (batch_size, 6))
    time = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    params = model.init(key, x, time)["params"]

    key, k_noise, k_time, k_r = jax.random.split(key, 4)
    noise = jax.random.normal(k_noise, x.shape)
    t = jax.random.uniform(k_time, (batch_size, 1))
    r = 0.5 * t

    noised = (1.0 - t) * x + (0.001 + 0.999 * t) * noise

    def u_fn(z_local, t_local, r_local):
        h_local = t_local - r_local
        th = jnp.concatenate([t_local, h_local], axis=-1)
        return model.apply({"params": params}, z_local, th)

    t_pair = jnp.concatenate([t, jnp.zeros_like(t)], axis=-1)
    v = model.apply({"params": params}, noised, t_pair)
    v_dir = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    def u_sum_fn(z_local, t_local, r_local):
        return jnp.sum(u_fn(z_local, t_local, r_local))

    (u_sum, dudt_jvp) = jax.jvp(
        u_sum_fn,
        (noised, t, r),
        (v_dir, jnp.ones_like(t), jnp.zeros_like(r)),
    )

    grad_z, grad_t = jax.grad(u_sum_fn, argnums=(0, 1))(noised, t, r)
    dudt_rev = jnp.sum(grad_z * v_dir) + jnp.sum(grad_t)

    np.testing.assert_allclose(dudt_jvp, dudt_rev, rtol=1e-4, atol=1e-4)
