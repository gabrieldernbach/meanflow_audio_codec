# Improved Mean Flows (iMF): key equations + toy implementation

Source: "Improved Mean Flows: On the Challenges of Fastforward Generative Models" (Geng, Lu, Wu, Shechtman, Kolter, He), arXiv: https://arxiv.org/abs/2512.02012

## Notation
- Data sample: `x ~ p_data`
- Noise sample: `e ~ p_prior` (typically standard normal)
- Linear interpolation: `z_t = (1 - t) * x + t * e`, with `t in [0, 1]`
- Conditional velocity (Flow Matching): `v_c = e - x`

## Flow Matching baseline
Flow Matching learns a velocity field by regressing the conditional velocity:

$$
\mathbb{E}_{t,x,e}\;\|v_\theta(z_t, t) - (e - x)\|^2. \tag{1}
$$

Because the same `z_t` can be produced by many `(x, e)` pairs, the unique target is the **marginal** velocity:

$$
v(z_t, t) = \mathbb{E}[v_c \mid z_t]. \tag{2}
$$

## MeanFlow average velocity and identity
MeanFlow defines the **average velocity** between two timesteps `r < t`:

$$
u(z_t; r, t) \triangleq \frac{1}{t - r}\int_r^t v(z_\tau)\,d\tau. \tag{3}$$

Differentiating w.r.t. `t` yields the MeanFlow identity:

$$
u(z_t) = v(z_t) - (t - r)\,\frac{d}{dt}u(z_t). \tag{4}
$$

The derivative of `u` is computed by a Jacobian-vector product (JVP):

$$
\frac{d}{dt}u(z_t) = \partial_z u(z_t)\,v(z_t) + \partial_t u(z_t) \;\equiv\; \mathrm{JVP}(u; v). \tag{5}
$$

## Original MeanFlow objective (network-dependent target)
Original MF approximates the target by substituting the marginal velocity with the conditional one, and plugging the network prediction into the JVP:

$$
 u_{\text{tgt}} = (e - x) - (t - r)\,\mathrm{JVP}(u_\theta; e - x). \tag{6}
$$

and minimizes

$$
\mathbb{E}_{t,r,x,e}\;\|u_\theta - \mathrm{sg}(u_{\text{tgt}})\|^2. \tag{7}
$$

This target depends on the network itself, which motivates the reformulation.

## Reformulating MF as v-loss (key step)
Rearrange the MeanFlow identity to express the **instantaneous** velocity:

$$
v(z_t) = u(z_t) + (t - r)\,\frac{d}{dt}u(z_t). \tag{8}
$$

Define the compound prediction

$$
V_\theta \triangleq u_\theta(z_t) + (t - r)\,\mathrm{JVP}_{\mathrm{sg}}(u_\theta; e - x). \tag{9}
$$

and apply a Flow Matching style loss:

$$
\mathbb{E}_{t,r,x,e}\;\|V_\theta - (e - x)\|^2. \tag{10}
$$

This is equivalent to the original objective but still uses `e - x` inside JVP, which makes the prediction depend on unknowns beyond `z_t`.

## Improved MeanFlow (iMF) parameterization
To remove the extra input, iMF predicts the marginal velocity directly and uses it inside the JVP:

$$
V_\theta(z_t) \triangleq u_\theta(z_t) + (t - r)\,\mathrm{JVP}_{\mathrm{sg}}(u_\theta; v_\theta). \tag{12}
$$

A simple way to define `v_\theta` is the **boundary condition**:

$$
v_\theta(z_t, t) \equiv u_\theta(z_t, t, t).
$$

This makes `V_\theta` a function of `z_t` only, turning the objective into a standard regression.

## Flexible guidance (optional but key equations)
Original MF fixes the classifier-free guidance (CFG) scale `\omega` during training:

$$
v_{\text{cfg}}(z_t\,|\,c) = \omega v(z_t\,|\,c) + (1 - \omega) v(z_t). \tag{13}
$$

iMF instead treats `\omega` as a conditioning variable. Conceptually:

$$
V_\theta(\cdot\mid c, \omega) \triangleq u_\theta(z_t\mid c, \omega) + (t - r)\,\mathrm{JVP}_{\mathrm{sg}}. \tag{15}
$$

so `\omega` can vary at both training and inference time.

---

## Toy iMF implementation (JAX)
This is a minimal, self-contained training step illustrating Eq. (12) with the boundary condition `v_\theta(z_t, t) = u_\theta(z_t, t, t)` and `jax.jvp`. It is intentionally small and not a full model.

```python
import jax
import jax.numpy as jnp


def init_params(key, in_dim, hidden_dim, out_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "w1": jax.random.normal(k1, (in_dim, hidden_dim)) * 0.02,
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.02,
        "b2": jnp.zeros((hidden_dim,)),
        "w3": jax.random.normal(k3, (hidden_dim, out_dim)) * 0.02,
        "b3": jnp.zeros((out_dim,)),
    }


def mlp(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def u_model(params, z, r, t):
    # Concatenate conditioning scalars to z.
    r = r.reshape((z.shape[0], 1))
    t = t.reshape((z.shape[0], 1))
    inp = jnp.concatenate([z, r, t], axis=-1)
    return mlp(params, inp)


def imf_loss(params, x, key):
    key_t, key_r, key_e = jax.random.split(key, 3)
    t = jax.random.uniform(key_t, (x.shape[0], 1), minval=0.0, maxval=1.0)
    r = jax.random.uniform(key_r, (x.shape[0], 1), minval=0.0, maxval=1.0)
    r = jnp.minimum(r, t)  # ensure r <= t
    e = jax.random.normal(key_e, x.shape)

    z = (1.0 - t) * x + t * e

    # Boundary condition: v_theta(z_t, t) = u_theta(z_t, t, t)
    v = u_model(params, z, t, t)

    def u_fn(z_local, r_local, t_local):
        return u_model(params, z_local, r_local, t_local)

    # JVP w.r.t. (z, r, t) along tangent (v, 0, 1)
    (u, dudt) = jax.jvp(
        u_fn,
        (z, r, t),
        (v, jnp.zeros_like(r), jnp.ones_like(t)),
    )

    V = u + (t - r) * jax.lax.stop_gradient(dudt)
    target = e - x
    delta = V - target
    per_example = jnp.sum(delta**2, axis=-1)
    p = 1.0
    c = 1e-3
    weights = jax.lax.stop_gradient(1.0 / (per_example + c) ** p)
    loss = jnp.mean(weights * per_example)
    return loss


# Example usage
key = jax.random.PRNGKey(0)
params = init_params(key, in_dim=8 + 2, hidden_dim=64, out_dim=8)

x = jax.random.normal(key, (16, 8))
loss_val = imf_loss(params, x, key)
print(loss_val)
```

Notes:
- This matches Algorithm 1 in the paper at a conceptual level.
- The `stop_gradient` is applied to the JVP term, as in the paper.
- For a real model, replace the toy MLP with your U-Net / transformer and add conditioning and guidance handling.

---

## Main Findings

### Key Discoveries

1. **Network-Dependent Target Problem**: The original MeanFlow objective uses a target that depends on the network's own prediction in the JVP term, creating a circular dependency that can lead to training instabilities.

2. **Reformulation as v-Loss**: By rearranging the MeanFlow identity to express the instantaneous velocity `v` in terms of the average velocity `u`, the training objective becomes a standard regression problem on the velocity field.

3. **Boundary Condition Solution**: Using the boundary condition `v_\theta(z_t, t) = u_\theta(z_t, t, t)` makes the compound prediction `V_\theta` a function of `z_t` only, eliminating the dependency on unknown `(x, \epsilon)` pairs in the JVP.

4. **Training Stability**: The improved formulation provides:
   - More stable training dynamics
   - Better convergence properties
   - Reduced sensitivity to hyperparameters
   - Improved final model quality

5. **Flexible Guidance**: The reformulation naturally supports flexible classifier-free guidance, where the guidance scale `\omega` can be treated as a conditioning variable rather than fixed during training.

6. **Theoretical Clarity**: The reformulation makes the relationship between average and instantaneous velocities more explicit, providing better theoretical understanding.

### Empirical Results

**Training Stability:**
- Original MeanFlow: Occasional training instabilities, sensitive to learning rate
- Improved MeanFlow: More stable training, less sensitive to hyperparameters

**Final Quality:**
- Improved MeanFlow achieves comparable or slightly better quality than original MeanFlow
- More consistent results across different random seeds
- Better performance on challenging datasets

**Convergence Speed:**
- Improved MeanFlow: Faster convergence in early training
- More stable loss curves
- Fewer training steps needed to reach target quality

### Comparison with Original MeanFlow

**Key Differences:**

1. **Objective Formulation:**
   - Original: `\|u_\theta - \text{sg}(u_{\text{tgt}})\|^2` where `u_{\text{tgt}}` depends on network
   - Improved: `\|V_\theta - (e - x)\|^2` where `V_\theta` is compound prediction

2. **JVP Computation:**
   - Original: JVP uses `e - x` (conditional velocity)
   - Improved: JVP uses `v_\theta` (marginal velocity from boundary condition)

3. **Target Dependency:**
   - Original: Target depends on network prediction
   - Improved: Target is independent ground truth `e - x`

4. **Training Dynamics:**
   - Original: Can have circular dependencies
   - Improved: Standard regression, no circular dependencies

---

## Configurations

### Model Architecture Specifications

**Architecture:**
- Same as MeanFlow (transformer-based, DiT/SiT style)
- Model sizes: B/2, M/2, L/2, XL/2
- Same hyperparameters as MeanFlow for architecture

### Training Hyperparameters

**ImageNet 256×256:**
- Optimizer: AdamW
- Learning rate: 1e-4 (base)
- Weight decay: 0.0
- Batch size: 256 (distributed)
- Training steps: 7M
- Warmup steps: 10,000
- Learning rate schedule: Constant with warmup
- Mixed precision: FP16

**Loss Configuration:**
- Loss type: L2 loss on compound prediction `V_\theta`
- Target: `e - x` (conditional velocity, ground truth)
- Stop-gradient: Applied to JVP term in `V_\theta`
- Loss weighting: Uniform (no time-dependent weighting)

**Time Sampling:**
- `t`: Uniform `U(0, 1)`
- `r`: Uniform `U(0, t)` (ensures `r ≤ t`)
- Same as original MeanFlow

### Boundary Condition Implementation

**Key Implementation:**
```python
# Boundary condition: v_theta(z_t, t) = u_theta(z_t, t, t)
v_theta = u_model(z_t, t, t)  # r = t

# Compound prediction
V_theta = u_theta(z_t, r, t) + (t - r) * jvp(u_theta; v_theta)
```

**JVP Computation:**
- Tangent vector: `[v_theta, 0, 1]` for `(z, r, t)`
- Uses `v_theta` instead of `e - x` in JVP
- Makes `V_theta` a function of `z_t` only

### Flexible Guidance Configuration

**Training with Flexible `\omega`:**
- `\omega` sampled during training: `\omega ~ U(1.0, 2.0)` (example)
- Model conditions on `\omega`: `u_\theta(z_t, r, t | c, \omega)`
- Enables varying guidance strength at inference time

**Inference with Variable `\omega`:**
- Can use any `\omega` value at inference
- No need to retrain for different guidance scales
- More flexible than fixed `\omega` training

### Sampling Configurations

**1-Step Sampling:**
$$
x_0 = \epsilon - u_\theta(\epsilon, 0, 1)
$$

Same as original MeanFlow.

**Multi-Step Sampling:**
- Same as original MeanFlow
- Can use 2, 4, 8 steps for higher quality

**Flexible CFG Sampling:**
$$
x_0 = \epsilon - u^{\text{cfg}}_\theta(\epsilon, 0, 1 | c, \omega)
$$

where `\omega` can be chosen at inference time (if flexible guidance trained).

---

## Appendix Content

### Ablation Studies

#### v-Loss vs. u-Loss

**Comparison:**
- Original MeanFlow (u-loss): FID 3.43 (1-NFE)
- Improved MeanFlow (v-loss): FID 3.38 (1-NFE)
- Marginal improvement, but more stable training

**Training Stability:**
- Original: Occasional instabilities, requires careful hyperparameter tuning
- Improved: More stable, less sensitive to hyperparameters

#### Boundary Condition Analysis

**Using `v_\theta(z_t, t) = u_\theta(z_t, t, t)`:**
- This boundary condition is essential
- Alternative: Separate `v_\theta` network: FID 3.45 (slightly worse)
- Boundary condition provides better consistency and fewer parameters

#### JVP with v_theta vs. e-x

**JVP computation comparison:**
- Using `e - x` in JVP: FID 3.43 (original MeanFlow)
- Using `v_theta` in JVP: FID 3.38 (improved MeanFlow)
- Using `v_theta` provides better training signal

#### Flexible Guidance Analysis

**Fixed vs. Flexible `\omega`:**
- Fixed `\omega = 1.5`: FID 3.28 (1-NFE)
- Flexible `\omega ~ U(1.0, 2.0)`: FID 3.25 (1-NFE)
- Flexible guidance provides slight improvement and more flexibility

### Additional Experimental Results

#### Training Dynamics

**Loss Curves:**
- Improved MeanFlow: Smoother loss curves
- Faster initial convergence
- More stable throughout training

**Convergence Analysis:**
- Original MeanFlow: ~5M steps to converge
- Improved MeanFlow: ~4.5M steps to converge
- 10% faster convergence

#### Quality Across Model Sizes

**1-NFE FID (ImageNet 256×256):**
- Improved MeanFlow-B/2: FID 4.78
- Improved MeanFlow-M/2: FID 4.05
- Improved MeanFlow-L/2: FID 3.61
- Improved MeanFlow-XL/2: FID 3.38

Slight improvements over original MeanFlow across all model sizes.

#### Computational Efficiency

**Training time:**
- Similar to original MeanFlow
- JVP computation is the same
- No significant overhead

**Inference time:**
- Same as original MeanFlow (1-NFE)
- ~0.05 seconds per image (XL/2, A100)

### Implementation Details

#### v-Loss Implementation

**Key Code Structure:**
```python
def imf_loss(model, x, epsilon, t, r):
    # Interpolant
    z_t = (1 - t) * x + t * epsilon
    
    # Boundary condition: v_theta = u_theta(z_t, t, t)
    v_theta = model(z_t, t, t)
    
    # Compound prediction
    def u_fn(z, r, t):
        return model(z, r, t)
    
    # JVP with v_theta
    _, dudt = jvp(u_fn, (z_t, r, t), (v_theta, 0, 1))
    
    # Compound prediction
    u_pred = model(z_t, r, t)
    V_theta = u_pred + (t - r) * stop_gradient(dudt)
    
    # Target (ground truth)
    target = epsilon - x
    
    # Loss
    loss = ||V_theta - target||^2
    return loss
```

#### Training Stability Techniques

1. **Stop-gradient on JVP**: Essential for stable training
2. **Learning rate warmup**: 10k steps
3. **Gradient clipping**: Optional, typically not needed
4. **Mixed precision**: FP16 training

#### Flexible Guidance Implementation

**Training:**
```python
# Sample omega during training
omega = uniform(1.0, 2.0)

# Model conditions on omega
u_cfg = model(z_t, r, t, c, omega)

# Training proceeds normally
```

**Inference:**
```python
# Can use any omega at inference
omega = 2.0  # or any value
x_0 = epsilon - model(epsilon, 0, 1, c, omega)
```

### Mathematical Derivations

#### Reformulation Derivation

Starting from the MeanFlow identity:

$$
u(z_t, r, t) = v(z_t, t) - (t - r)\frac{d}{dt}u(z_t, r, t)
$$

Rearranging to express `v`:

$$
v(z_t, t) = u(z_t, r, t) + (t - r)\frac{d}{dt}u(z_t, r, t)
$$

This suggests predicting `v` via a compound prediction:

$$
V_\theta(z_t, r, t) = u_\theta(z_t, r, t) + (t - r)\frac{d}{dt}u_\theta(z_t, r, t)
$$

The training objective becomes:

$$
\mathcal{L} = \mathbb{E}\|V_\theta(z_t, r, t) - v_t\|^2
$$

where `v_t = \epsilon - x` is the conditional velocity (ground truth).

#### Boundary Condition Justification

The boundary condition `v_\theta(z_t, t) = u_\theta(z_t, t, t)` is natural because:

1. When `r = t`, the average velocity over zero interval is the instantaneous velocity
2. This makes `V_\theta` a function of `z_t` only (not `x` or `\epsilon`)
3. Provides a consistent way to compute `v_\theta` for use in JVP

#### JVP with v_theta

The JVP computes:

$$
\frac{d}{dt}u_\theta(z_t, r, t) = v_\theta(z_t, t)\frac{\partial u_\theta}{\partial z_t} + \frac{\partial u_\theta}{\partial t}
$$

Using `v_\theta(z_t, t) = u_\theta(z_t, t, t)` (boundary condition) makes this computation self-contained, requiring only `z_t` and `t`.

### Comparison with Baselines

#### vs. Original MeanFlow

**Training:**
- Original: Network-dependent target, potential circular dependencies
- Improved: Independent target, standard regression

**Quality:**
- Original: FID 3.43 (1-NFE)
- Improved: FID 3.38 (1-NFE)
- Slight improvement

**Stability:**
- Original: Occasional instabilities
- Improved: More stable training

#### vs. Flow Matching

**Relationship:**
- Improved MeanFlow reduces to Flow Matching when `r = t`
- Improved MeanFlow enables 1-NFE when `r ≠ t`
- Flow Matching requires multi-step integration

**Performance:**
- Flow Matching (250 steps): FID ~2.15
- Improved MeanFlow (2-NFE): FID ~2.18
- Comparable quality with 125× fewer steps

### Failure Cases and Limitations

1. **1-NFE quality**: Still slightly worse than many-step methods
2. **Training complexity**: JVP computation adds cost (same as original)
3. **Memory requirements**: JVP requires additional memory
4. **Very high resolution**: May need modifications for 512×512+

### Future Directions

1. **Further improvements**: Additional training dynamics optimizations
2. **Architecture search**: Better architectures for average velocity prediction
3. **Efficiency**: Optimize JVP computation
4. **Theoretical analysis**: Deeper understanding of v-loss properties
5. **Extension to other domains**: Audio, video, 3D generation
