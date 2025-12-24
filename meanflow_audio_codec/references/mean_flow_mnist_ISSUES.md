# Issues Found in mean_flow_mnist.py Reference Implementation

After comparing with the official JAX implementation and PyTorch implementation, several issues were identified:

## Critical Issues

### 1. **Incorrect JVP Call** (Line 72-73)
**Problem**: The JVP call only passes `(z, t)` but the forward function uses `r`. 

**Current (INCORRECT)**:
```python
def f(z_, t_): 
    return self.forward(z_, t_.unsqueeze(1), r.unsqueeze(1), cls_idx)
    
u, dudt = torch.autograd.functional.jvp(
    f, (z, t), (v, torch.ones_like(t)), create_graph=True)
```

**Should be**:
```python
def f(z_, t_, r_): 
    return self.forward(z_, t_.unsqueeze(1), r_.unsqueeze(1), cls_idx)
    
u, dudt = torch.autograd.functional.jvp(
    f, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)), create_graph=True)
```

The JVP must include all inputs that vary. Since `r` is used in forward but doesn't change in the JVP direction (dr/dt = 0), we pass `torch.zeros_like(r)` as the direction for `r`.

**Reference**: 
- Official JAX (meanflow.py:231): `jax.jvp(u_fn, (z_t, t, r), (v_g, dt_dt, dr_dt))` where `dr_dt = jnp.zeros_like(t)`
- PyTorch (meanflow.py:164): `jvp_args = (lambda z, t, r: model_partial(z, t, r), (z, t, r), (v_hat, torch.ones_like(t), torch.zeros_like(r)))`

### 2. **Time Distribution**
**Problem**: Uses uniform distribution instead of logit-normal distribution.

**Current**: `t = torch.rand(B, device=x0.device)` (uniform [0, 1])

**Official uses**: Logit-normal distribution with μ=-0.4, σ=1.0
```python
# Apply sigmoid to normal samples
normal_samples = np.random.randn(batch_size, 2) * sigma + mu
samples = 1 / (1 + np.exp(-normal_samples))
```

This is less critical but different from the paper's default.

## Minor Issues / Inconsistencies

### 3. **Adaptive Loss Implementation**
The adaptive loss formula appears correct, but verification:
- Reference: `w = 1 / (delta_sq + c).pow(1-gamma)` with `gamma=0.5` → `p = 0.5`
- Official JAX: Uses `norm_p = 1.0` by default (different default)
- PyTorch: Same formula as reference

The formula is correct, just different defaults.

### 4. **Missing Clip in u_tgt**
Official JAX clips: `u_tgt = v_g - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt`
Reference doesn't clip. This may be fine if t-r is always in [0,1] due to sampling, but adds robustness.

## Summary

The **critical issue is #1** - the JVP call is incorrect and will produce wrong gradients. Issue #2 (time distribution) is less critical but worth fixing for consistency with the paper. The adaptive loss formula (#3) appears correct.

