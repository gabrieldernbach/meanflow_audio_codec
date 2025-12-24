import jax
import jax.numpy as jnp


def sample(
    apply_fn,
    noise_dimension: int,
    params,
    key,
    latents: jnp.ndarray | None = None,
    n_steps: int = 100,
    use_improved_mean_flow: bool = False,
    guidance_scale: float = 1.0,
) -> jnp.ndarray:
    """Sample from flow model via numerical ODE integration.
    
    Integrates dx/dt = v(x, t) from t=1 to t=0 using Heun's method, where
    x(1) ~ N(0,I) is noise and x(0) is the generated sample.
    
    Supports classifier-free guidance (CFG) when guidance_scale > 1.0:
    v^cfg = guidance_scale * v(x, t | latents) + (1 - guidance_scale) * v(x, t)
    
    Args:
        apply_fn: Flow model forward function.
        noise_dimension: Dimension of noise vector.
        params: Model parameters.
        key: JAX random key.
        latents: Optional latent vector from encoder of shape [B, latent_dim].
                 If None, unconditional mode. Required if guidance_scale != 1.0.
        n_steps: Number of integration steps.
        use_improved_mean_flow: Whether to use improved mean flow (unused).
        guidance_scale: Classifier-free guidance scale. 
                       If 1.0, uses conditional prediction only.
                       If > 1.0, blends conditional and unconditional: 
                       pred = guidance_scale * pred_cond + (1 - guidance_scale) * pred_uncond
        
    Returns:
        Generated samples, shape [B, noise_dimension] where B is batch size from latents.
    """
    # Infer batch size from latents
    # For CFG, we always need latents to compute conditional prediction
    if latents is None:
        if guidance_scale != 1.0:
            raise ValueError("guidance_scale != 1.0 requires latents to be provided")
        raise ValueError("latents must be provided for conditional sampling")
    
    batch_size = latents.shape[0]
    
    key, k_init = jax.random.split(key)
    x = jax.random.normal(k_init, (batch_size, noise_dimension), dtype=jnp.float32)

    dt = 1.0 / float(n_steps)
    ts = jnp.linspace(1.0, 0.0, n_steps, dtype=jnp.float32)

    @jax.jit
    def heun_step(params, dt, x, t, latents, guidance_scale):
        # t: scalar float
        b = x.shape[0]
        t1 = jnp.full((b, 1), t, dtype=x.dtype)
        t1_pair = jnp.concatenate([t1, jnp.zeros_like(t1)], axis=-1)
        
        if guidance_scale == 1.0:
            # No CFG, just use conditional
            k1_cond = apply_fn({"params": params}, x, t1_pair, latents)
            k1 = k1_cond
        else:
            # CFG: blend conditional and unconditional
            k1_cond = apply_fn({"params": params}, x, t1_pair, latents)
            k1_uncond = apply_fn({"params": params}, x, t1_pair, None)
            k1 = guidance_scale * k1_cond + (1.0 - guidance_scale) * k1_uncond

        t2 = jnp.full((b, 1), t - dt, dtype=x.dtype)
        t2_pair = jnp.concatenate([t2, jnp.zeros_like(t2)], axis=-1)
        
        if guidance_scale == 1.0:
            k2_cond = apply_fn({"params": params}, x - dt * k1, t2_pair, latents)
            k2 = k2_cond
        else:
            # CFG: blend conditional and unconditional
            k2_cond = apply_fn({"params": params}, x - dt * k1, t2_pair, latents)
            k2_uncond = apply_fn({"params": params}, x - dt * k1, t2_pair, None)
            k2 = guidance_scale * k2_cond + (1.0 - guidance_scale) * k2_uncond

        x = x - (dt / 2.0) * (k1 + k2)
        return x

    def body(carry, t):
        x = carry
        x = heun_step(params, dt, x, t, latents, guidance_scale)
        return x, None

    # jax.lax.scan repeatedly applies the 'body' function over the time steps 'ts',
    # carrying forward the state 'x' to perform iterative denoising steps.
    x, _ = jax.lax.scan(body, x, ts)
    return x


