import jax
import jax.numpy as jnp


def sinusoidal_embedding(x: jnp.ndarray, dim: int, max_period: float = 10000.0) -> jnp.ndarray:
    """
    x: [B] in [0, 1]
    returns: [B, dim]
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = x[:, None].astype(jnp.float32) * freqs[None]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def weighted_l2_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    p: float = 1.0,
    c: float = 1e-3,
) -> jnp.ndarray:
    delta = pred - target
    per_example = jnp.sum(delta**2, axis=tuple(range(1, delta.ndim)))
    weights = jax.lax.stop_gradient(1.0 / (per_example + c) ** p)
    return jnp.mean(weights * per_example)


def ema(mu, dx, beta=0.99):
    return beta * mu + (1.0 - beta) * dx if mu is not None else dx


def logit_normal(key, shape, mean=-0.4, std=1.0, dtype=jnp.float32):
    return jax.nn.sigmoid(jax.random.normal(key, shape, dtype=dtype) * std + mean)


def sample_tr(key, batch_size: int, dtype=jnp.float32, mean=-0.4, std=1.0, data_proportion=0.5):
    k_t, k_r = jax.random.split(key, 2)
    t = logit_normal(k_t, (batch_size, 1), mean=mean, std=std, dtype=dtype)
    r = logit_normal(k_r, (batch_size, 1), mean=mean, std=std, dtype=dtype)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    data_size = int(batch_size * data_proportion)
    mask = jnp.arange(batch_size) < data_size
    mask = mask[:, None]
    r = jnp.where(mask, t, r)
    return t, r

