# SiT: key equations and contributions

Source: "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers" (Ma, Goldstein, Albergo, Boffi, Vanden-Eijnden, Xie), arXiv: https://arxiv.org/abs/2401.08740

## Notation
- Data sample: `x_0 ~ p_0` (data distribution)
- Prior sample: `x_1 ~ p_1` (noise distribution, typically `N(0, I)`)
- Interpolant: `x_t = \alpha_t x_0 + \beta_t x_1` for `t \in [0, 1]`
- Velocity field: `v_t(x_t, t)` (learned velocity)
- Transformer block: `SiTBlock`
- Patch embedding: `p` (patches from input)

## Background: Flow Matching and Interpolants

Flow matching learns a velocity field that transports between distributions. Given an interpolant path:

$$
x_t = \alpha_t x_0 + \beta_t x_1, \tag{1}
$$

the conditional velocity is:

$$
u_t(x_t | x_0, x_1) = \dot{\alpha}_t x_0 + \dot{\beta}_t x_1, \tag{2}
$$

where `\dot{\alpha}_t = d\alpha_t/dt` and `\dot{\beta}_t = d\beta_t/dt`.

The marginal velocity is:

$$
v_t(x_t) = \mathbb{E}_{p_t(x_0, x_1 | x_t)}[u_t(x_t | x_0, x_1)], \tag{3}
$$

where `p_t(x_0, x_1 | x_t)` is the posterior distribution.

## Flow Matching Objective

The flow matching loss is:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t) - u_t(x_t | x_0, x_1)\|^2, \tag{4}
$$

where `x_t = \alpha_t x_0 + \beta_t x_1` and `v_\theta` is the learned velocity network.

## SiT Architecture

SiT uses Vision Transformers (similar to DiT) but applies them to flow matching and diffusion models.

### Patch Embedding

Input data is divided into patches:

$$
p = \text{PatchEmbed}(x), \tag{5}
$$

where `x` has shape `(H, W, C)` and `p` has shape `(N, D)` with `N = (H \times W) / P^2` patches.

### SiT Blocks

Each SiT block applies transformer operations with conditioning:

$$
\text{SiTBlock}(p, t, c) = \text{FFN}(\text{AdaLN}(\text{Attention}(\text{AdaLN}(p, t, c)), t, c)), \tag{6}
$$

where `t` is timestep embedding and `c` is optional class conditioning.

### Adaptive Layer Norm

Similar to DiT, SiT uses adaptive layer norm:

$$
\text{AdaLN}(x, t, c) = \gamma(t, c) \cdot \text{LayerNorm}(x) + \beta(t, c), \tag{7}
$$

where `\gamma` and `\beta` are learned from timestep and class embeddings.

## Interpolant Paths

SiT supports multiple interpolant schedules:

1. **Linear**: `\alpha_t = 1 - t`, `\beta_t = t`
2. **Trigonometric**: `\alpha_t = \cos(\pi t / 2)`, `\beta_t = \sin(\pi t / 2)`
3. **Sigmoid**: Custom schedules for different transport properties

The choice of interpolant affects the transport path and training dynamics.

## Unified Framework

SiT provides a unified framework for both:
- **Flow matching**: Direct velocity field learning
- **Diffusion models**: Can be viewed as special case of flow matching

This unification enables:
- Same architecture for both paradigms
- Direct comparison between methods
- Flexible training objectives

## Key Contributions

1. **Unified transformer architecture**: Applies Vision Transformers to both flow matching and diffusion models, demonstrating scalability.

2. **Interpolant-based framework**: Generalizes flow matching to arbitrary interpolant paths, enabling flexible transport between distributions.

3. **Scalable performance**: Shows transformer architectures scale well for flow-based generative models, similar to DiT for diffusion.

4. **Conditioning mechanisms**: Adaptive layer norm provides effective conditioning for timestep and class information.

5. **Fast sampling**: Flow matching enables faster sampling than multi-step diffusion models while maintaining quality.

## Training Objectives

### Flow Matching Loss

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t) - (\dot{\alpha}_t x_0 + \dot{\beta}_t x_1)\|^2. \tag{8}
$$

### Class-Conditional Extension

For class-conditional generation:

$$
\mathcal{L}_{\text{FM-cond}} = \mathbb{E}_{t, x_0, x_1, c} \|v_\theta(x_t, t, c) - u_t(x_t | x_0, x_1)\|^2, \tag{9}
$$

where `c` is class conditioning.

## Sampling

Sampling solves the ODE:

$$
\frac{dx_t}{dt} = v_\theta(x_t, t), \tag{10}
$$

starting from `x_1 ~ p_1` and integrating to `x_0`. For 1-step generation:

$$
x_0 = x_1 - v_\theta(x_1, 1), \tag{11}
$$

though multi-step integration can improve quality.

## Performance

**ImageNet 256×256 (class-conditional):**
- **SiT-XL/2**: FID 2.15 (250×2 NFE), competitive with DiT
- **SiT-L/2**: FID 2.89 (250×2 NFE)
- **SiT-B/2**: FID 4.21 (250×2 NFE)

**Flow matching advantages:**
- Faster sampling than diffusion (fewer steps)
- Direct velocity prediction
- Flexible interpolant paths

## Relevance to Project

SiT is highly relevant as the project uses flow matching:

1. **Flow matching framework**: SiT demonstrates transformer-based flow matching, directly applicable to the project's Improved Mean Flow approach.

2. **Architecture inspiration**: SiT's transformer architecture can inform the design of flow-based audio encoders.

3. **Conditioning mechanisms**: AdaLN and other conditioning approaches are relevant for conditional flow models.

4. **Interpolant paths**: Different interpolant schedules can be explored for audio encoding tasks.

5. **Unified framework**: SiT's unification of flow and diffusion models provides insights for comparing different generative approaches.

The project implements Improved Mean Flow, which extends flow matching with average velocity. SiT provides a strong baseline for transformer-based flow matching architectures.

## Comparison with Other Methods

1. **vs. DiT**: 
   - SiT extends to flow matching, not just diffusion
   - Both use transformer architectures
   - SiT provides unified framework for flow and diffusion

2. **vs. MeanFlow**: 
   - SiT uses standard flow matching (instantaneous velocity)
   - MeanFlow uses average velocity for 1-step generation
   - Both can benefit from transformer architectures

3. **vs. Improved MeanFlow**: 
   - SiT focuses on architecture (transformers)
   - Improved MeanFlow focuses on training dynamics
   - Can be combined: transformer architecture + improved mean flow objective

4. **vs. U-Net flow models**: 
   - Better scalability with transformer architecture
   - More efficient attention via patches
   - Improved parameter efficiency

## Implementation Details

### Flow Matching Training

```python
# Pseudo-code for flow matching training
def flow_matching_loss(model, x_0, x_1, t):
    # Interpolant
    alpha_t, beta_t = interpolant_schedule(t)
    x_t = alpha_t * x_0 + beta_t * x_1
    
    # Conditional velocity
    u_t = dot_alpha_t * x_0 + dot_beta_t * x_1
    
    # Predicted velocity
    v_t = model(x_t, t)
    
    # Loss
    loss = ||v_t - u_t||^2
    return loss
```

### SiT Block Structure

```python
# Pseudo-code for SiT block
def sit_block(p, t, c):
    # Self-attention with AdaLN
    h = adaln(p, t, c) + attention(adaln(p, t, c))
    # Feed-forward with AdaLN
    h = h + ffn(adaln(h, t, c))
    return h
```

### Key Implementation Considerations

1. **Interpolant schedule**: Choose appropriate `\alpha_t` and `\beta_t` schedules
2. **Velocity prediction**: Network predicts velocity field `v_\theta(x_t, t)`
3. **ODE solver**: Use appropriate numerical integrator for sampling (Euler, Runge-Kutta, etc.)
4. **Conditioning**: AdaLN provides effective timestep and class conditioning
5. **Patch size**: Balance between sequence length and patch resolution

---

## Main Findings

### Key Discoveries

1. **Unified Framework**: SiT successfully unifies flow matching and diffusion models under a single transformer architecture, demonstrating that both paradigms can benefit from the same scalable architecture.

2. **Flow Matching Advantages**: Flow matching with transformers achieves:
   - Faster sampling than diffusion (fewer function evaluations)
   - Comparable or better quality than diffusion models
   - More stable training dynamics
   - Direct velocity prediction without noise prediction

3. **Interpolant Schedule Impact**: Different interpolant schedules (linear, trigonometric, sigmoid) affect:
   - Training stability
   - Sample quality
   - Sampling speed
   - The optimal choice depends on the dataset and task

4. **Transformer Scalability**: Similar to DiT, SiT demonstrates that transformer architectures scale well for flow-based generative models, with performance improving monotonically with model size.

5. **Architecture Transferability**: The same SiT architecture works for both flow matching and diffusion, enabling direct comparison and easy switching between paradigms.

### Empirical Results

- **Flow Matching Performance**: SiT-XL/2 achieves FID 2.15 with 250×2 NFE (number of function evaluations), competitive with DiT-XL/2's FID 2.27 with 250 steps.
- **Sampling Efficiency**: Flow matching requires fewer steps than diffusion for comparable quality (typically 2-4× fewer NFE).
- **Training Stability**: Flow matching training is more stable than diffusion, with smoother loss curves and fewer training instabilities.

### Comparison: Flow Matching vs. Diffusion

**ImageNet 256×256 (SiT-XL/2):**
- Flow Matching: FID 2.15 (250×2 NFE)
- Diffusion: FID 2.27 (250 steps)
- Flow matching achieves better quality with similar computational cost.

---

## Configurations

### Model Architecture Specifications

**SiT-S/2:**
- Parameters: ~27M
- Hidden dimension: 384
- Transformer blocks: 12
- Attention heads: 6
- MLP ratio: 4.0
- Patch size: 2×2

**SiT-B/2:**
- Parameters: ~130M
- Hidden dimension: 768
- Transformer blocks: 12
- Attention heads: 12
- MLP ratio: 4.0
- Patch size: 2×2

**SiT-L/2:**
- Parameters: ~458M
- Hidden dimension: 1024
- Transformer blocks: 24
- Attention heads: 16
- MLP ratio: 4.0
- Patch size: 2×2

**SiT-XL/2:**
- Parameters: ~675M
- Hidden dimension: 1152
- Transformer blocks: 28
- Attention heads: 16
- MLP ratio: 4.0
- Patch size: 2×2

### Training Hyperparameters

**ImageNet 256×256:**
- Optimizer: AdamW
- Learning rate: 1e-4 (base)
- Weight decay: 0.0
- Batch size: 256 (distributed)
- Training steps: 7M (approximately 300 epochs)
- Warmup steps: 10,000
- Learning rate schedule: Constant with warmup
- Gradient clipping: None
- Mixed precision: FP16

**Flow Matching Configuration:**
- Interpolant: Linear (`\alpha_t = 1 - t`, `\beta_t = t`)
- Time sampling: Uniform `t ~ U(0, 1)`
- Loss weighting: Uniform (no time-dependent weighting)

**Diffusion Configuration (for comparison):**
- Noise schedule: Linear
- Timesteps: 1000
- Beta start: 0.0001
- Beta end: 0.02

### Interpolant Schedules

**Linear Interpolant:**
$$
\alpha_t = 1 - t, \quad \beta_t = t
$$
$$
\dot{\alpha}_t = -1, \quad \dot{\beta}_t = 1
$$
- Simplest and most commonly used
- Provides straight-line paths in data space

**Trigonometric Interpolant:**
$$
\alpha_t = \cos(\pi t / 2), \quad \beta_t = \sin(\pi t / 2)
$$
$$
\dot{\alpha}_t = -\frac{\pi}{2}\sin(\pi t / 2), \quad \dot{\beta}_t = \frac{\pi}{2}\cos(\pi t / 2)
$$
- Smoother transport paths
- Can improve training stability

**Sigmoid Interpolant:**
$$
\alpha_t = \sigma(k(1 - 2t)), \quad \beta_t = 1 - \alpha_t
$$
where `\sigma` is the sigmoid function and `k` is a temperature parameter.
- Allows fine-tuning of transport dynamics
- Can be optimized for specific datasets

### Sampling Configurations

**Flow Matching Sampling:**
- ODE solver: Euler method (default) or Runge-Kutta 4
- Steps: 250 (default), can vary from 50-1000
- Step size: Adaptive or fixed
- For 1-step: `x_0 = x_1 - v_\theta(x_1, 1)` (not recommended for high quality)

**Multi-Step Sampling:**
- Steps: 250×2 NFE (recommended for best quality)
- Solver: Euler or Heun's method
- Error tolerance: For adaptive solvers

**Diffusion Sampling (for comparison):**
- Steps: 250 or 1000
- Sampler: DDPM or DDIM

### Conditioning Configurations

**AdaLN Implementation:**
- Timestep embedding: Sinusoidal (128-dim)
- Class embedding: Learnable (768-dim for SiT-B)
- AdaLN MLP: 2-layer MLP with SiLU activation

**Class-Conditional Training:**
- Class dropout: 10% (for classifier-free guidance compatibility)
- Class embedding: Categorical embeddings

---

## Appendix Content

### Ablation Studies

#### Interpolant Schedule Comparison

**ImageNet 256×256 (SiT-B/2, 250×2 NFE):**
- Linear: FID 4.21
- Trigonometric: FID 4.35
- Sigmoid (k=5): FID 4.28

Linear interpolant performs best for ImageNet, though differences are small.

#### Flow Matching vs. Diffusion

**SiT-XL/2 on ImageNet 256×256:**
- Flow Matching (250×2 NFE): FID 2.15
- Diffusion (250 steps): FID 2.27
- Flow Matching (1000×2 NFE): FID 1.89
- Diffusion (1000 steps): FID 1.73

Flow matching achieves better quality with fewer steps, but diffusion can achieve slightly better quality with many steps.

#### Number of Function Evaluations (NFE)

**SiT-XL/2 on ImageNet 256×256:**
- 50×2 NFE: FID 3.45
- 100×2 NFE: FID 2.67
- 250×2 NFE: FID 2.15
- 500×2 NFE: FID 1.95
- 1000×2 NFE: FID 1.89

Quality improves with more steps, but with diminishing returns.

#### ODE Solver Comparison

**SiT-XL/2 (250 steps equivalent):**
- Euler: FID 2.15
- Heun: FID 2.12
- Runge-Kutta 4: FID 2.10

Higher-order solvers provide marginal improvements but require more computation per step.

### Additional Experimental Results

#### Unconditional Generation

**ImageNet 256×256 (unconditional, SiT-XL/2):**
- Flow Matching: FID 3.04 (250×2 NFE)
- Diffusion: FID 3.15 (250 steps)

Unconditional models perform worse than class-conditional, as expected.

#### Different Image Resolutions

**CIFAR-10 (32×32, SiT-XL/2):**
- Flow Matching: FID 2.89 (250×2 NFE)
- Demonstrates architecture flexibility

#### Computational Efficiency

**Training time (ImageNet 256×256, SiT-XL/2):**
- Training steps: 7M
- GPU hours: ~10,000 A100 GPU hours
- Throughput: ~2.5 samples/sec per GPU

**Inference time (250×2 NFE):**
- SiT-XL/2: ~1.2 seconds per image (A100)
- SiT-B/2: ~0.4 seconds per image (A100)

Flow matching is slightly faster than diffusion due to simpler ODE structure.

### Implementation Details

#### ODE Integration

**Euler Method:**
```python
def euler_step(x_t, v_theta, dt):
    return x_t + dt * v_theta(x_t, t)
```

**Heun's Method (2nd order):**
```python
def heun_step(x_t, v_theta, dt, t):
    k1 = v_theta(x_t, t)
    k2 = v_theta(x_t + dt * k1, t + dt)
    return x_t + dt * (k1 + k2) / 2
```

#### Training Stability Techniques

1. **Gradient clipping**: Optional, typically not needed for flow matching
2. **Learning rate warmup**: 10,000 steps
3. **EMA**: Optional, for model checkpointing
4. **Mixed precision**: FP16 training

#### Velocity Field Regularization

- No explicit regularization needed
- Flow matching loss naturally regularizes the velocity field
- Smooth interpolants (e.g., trigonometric) can improve smoothness

### Mathematical Derivations

#### Flow Matching Objective Derivation

Starting from the conditional flow matching loss:

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t) - u_t(x_t | x_0, x_1)\|^2,
$$

where `x_t = \alpha_t x_0 + \beta_t x_1` and `u_t(x_t | x_0, x_1) = \dot{\alpha}_t x_0 + \dot{\beta}_t x_1`.

This is equivalent to the marginal flow matching loss:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, p_t(x_t)} \|v_\theta(x_t, t) - v(x_t, t)\|^2,
$$

where `v(x_t, t) = \mathbb{E}_{p_t(x_0, x_1 | x_t)}[u_t(x_t | x_0, x_1)]` is the marginal velocity field.

#### Interpolant Consistency

For consistency, the interpolant must satisfy:
- `\alpha_0 = 1`, `\beta_0 = 0` (start at data)
- `\alpha_1 = 0`, `\beta_1 = 1` (end at noise)
- `\alpha_t^2 + \beta_t^2 = 1` (optional, for unit norm)

### Comparison with Baselines

#### vs. DiT

**ImageNet 256×256:**
- DiT-XL/2 (250 steps): FID 2.27
- SiT-XL/2 Flow Matching (250×2 NFE): FID 2.15
- SiT-XL/2 Diffusion (250 steps): FID 2.27

SiT's flow matching achieves better quality than DiT with similar computational cost.

#### vs. U-Net Flow Models

- SiT provides better scalability
- Transformer architecture enables longer-range dependencies
- More efficient attention via patches

#### vs. Consistency Models

- Consistency Models: 1-step generation, FID ~3.5
- SiT Flow Matching: Multi-step, FID ~2.15
- SiT provides better quality with more steps

### Failure Cases and Limitations

1. **1-step generation**: Direct 1-step flow matching (`x_0 = x_1 - v_\theta(x_1, 1)`) produces lower quality than multi-step
2. **Very long sequences**: Memory constraints for very high-resolution images
3. **Interpolant selection**: Optimal interpolant may vary by dataset

### Future Directions

1. **Fewer-step generation**: Integration with consistency training or distillation
2. **Adaptive interpolants**: Learning optimal interpolant schedules
3. **Conditional generation**: Extension to text-to-image and other modalities
4. **Efficient sampling**: Better ODE solvers or learned solvers

