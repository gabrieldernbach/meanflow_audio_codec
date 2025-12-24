# Method Comparison: Flow Matching vs Mean Flow vs Improved Mean Flow

This document summarizes the key differences between the three flow-based generative modeling methods implemented in this directory.

## Overview

All three methods use the same architecture (`ConditionalFlow` with residual blocks) but differ in:
- **Loss function formulation**
- **Forward pass inputs** (time embeddings)
- **Training objectives**
- **Sampling efficiency**

## 1. Flow Matching (`flow.py`)

**Method**: Standard conditional Flow Matching (Lipman et al., 2023)

### Key Characteristics
- **Inputs**: `(x, t, cls_idx)` - single time embedding
- **Loss**: `flow_matching_loss()` - Direct regression on conditional velocity
- **Target**: `v_θ(z_t, t) ≈ v_c = (e - x)` where `z_t = (1-t)x + t·e`
- **Noise schedule**: Linear interpolation with `noise_min` and `noise_max` parameters
- **Training**: Standard MSE loss on velocity prediction

### Loss Function
```python
time = random_t ∈ [0, 1]
noised = (1-time) * x + (noise_min + noise_max*time) * noise
target = noise * noise_max - x
loss = MSE(model(noised, time, cls_idx), target)
```

### Sampling
- Requires **~100 steps** for high quality (Heun's method ODE solver)
- Each step requires one forward pass

### Advantages
- Simple, stable training
- Well-established baseline

### Disadvantages
- Slow generation (high NFE - Number of Function Evaluations)
- Not suitable for real-time applications

---

## 2. Mean Flow (`mean_flow.py`)

**Method**: Mean Flow (Geng et al., 2025)

### Key Characteristics
- **Inputs**: `(x, t, r, cls_idx)` - dual time embeddings with `r ≤ t`
- **Loss**: `mean_flow_loss()` - Average velocity with adaptive reweighting
- **Concept**: Models **average velocity** `u(z_t, r, t)` between timesteps `r` and `t`
- **Training**: Adaptive reweighting based on prediction error

### Loss Function
```python
t, r = random_times where r ≤ t
flow_ratio fraction: set r = t (reduces to Flow Matching)
z = (1-t) * x + t * e
u = model(z, t, r, cls_idx)
u_target = (e - x) - (t-r) * JVP(model, (z, t, r), (e-x, 1, 0))
error = u - detach(u_target)
weight = 1 / (error² + c)^(1-γ)
loss = mean(weight * error²)
```

### Key Hyperparameters
- `gamma = 0.5`: Reweighting exponent
- `c = 1e-3`: Stability constant
- `flow_ratio = 0.5`: Fraction of samples with `r = t`

### Sampling
- Typically **2-5 steps** for good quality
- Significant speedup vs Flow Matching (20-50×)

### Advantages
- Fast generation (low NFE)
- Adaptive reweighting helps with difficult samples

### Disadvantages
- Network-dependent target (uses `u_θ` in JVP)
- More complex training objective

---

## 3. Improved Mean Flow (`improved_mean_flow.py`)

**Method**: Improved Mean Flow / iMF (Geng et al., 2024)

### Key Characteristics
- **Inputs**: `(x, t, r, cls_idx)` - dual time embeddings with `r ≤ t`
- **Loss**: `improved_mean_flow_loss()` - v-loss formulation
- **Concept**: Reformulates Mean Flow to eliminate network-dependent target
- **Training**: Standard regression (no adaptive weighting)

### Key Differences from Mean Flow

1. **Boundary Condition**: `v_θ(z_t, t) = u_θ(z_t, t, t)` (explicit velocity prediction)

2. **JVP Direction**: Uses `v_θ` instead of `(e - x)` in JVP computation
   ```python
   v_theta = model(z, t, t, cls_idx)  # Boundary condition
   JVP uses tangent vector: (v_theta, 0, 1) instead of (e-x, 1, 0)
   ```

3. **Compound Prediction**: `V_θ = u_θ + (t-r) · detach(JVP)`

4. **Loss**: Standard L2 regression (no adaptive weighting)
   ```python
   V_theta = u + (t-r) * detach(dudt)
   target = e - x
   loss = MSE(V_theta, target)
   ```

### Loss Function
```python
t, r = random_times where r ≤ t
flow_ratio fraction: set r = t (reduces to Flow Matching)
z = (1-t) * x + t * e

# Boundary condition: explicit velocity
v_theta = model(z, t, t, cls_idx)

# JVP along (v_theta, 0, 1)
u, dudt = JVP(model, (z, t, r), (v_theta, 0, 1))

# Compound prediction
V_theta = u + (t-r) * detach(dudt)

# Standard regression
loss = MSE(V_theta, e - x)
```

### Key Hyperparameters
- `flow_ratio = 0.5`: Fraction of samples with `r = t`
- **No gamma or c parameters** (simpler than Mean Flow)

### Sampling
- Typically **2-5 steps** (similar to Mean Flow)
- Same speedup benefits as Mean Flow

### Advantages
- **Network-independent target** (eliminates training instability from Mean Flow)
- Simpler objective (no adaptive weighting)
- More stable training dynamics
- Same fast generation as Mean Flow

### Disadvantages
- Slightly more complex forward pass (boundary condition computation)

---

## Comparison Table

| Aspect | Flow Matching | Mean Flow | Improved Mean Flow |
|--------|--------------|-----------|-------------------|
| **Time Inputs** | `t` only | `t, r` (r ≤ t) | `t, r` (r ≤ t) |
| **Loss Type** | Direct regression | Adaptive weighted | Standard regression |
| **Target Formulation** | `e - x` | Network-dependent | Network-independent |
| **JVP Direction** | N/A | `(e-x, 1, 0)` | `(v_θ, 0, 1)` |
| **Boundary Condition** | N/A | Implicit | Explicit `v_θ = u_θ(t,t)` |
| **Reweighting** | No | Yes (gamma, c) | No |
| **Sampling Steps** | ~100 | 2-5 | 2-5 |
| **Speedup vs FM** | 1× | 20-50× | 20-50× |
| **Training Stability** | High | Medium | High |
| **Implementation Complexity** | Low | Medium | Medium |

---

## Method Progression

```
Flow Matching (baseline)
    ↓
Mean Flow (introduces average velocity, adaptive weighting)
    ↓
Improved Mean Flow (eliminates network-dependent target, stabilizes training)
```

## When to Use Each Method

- **Flow Matching**: Baseline comparison, when sampling speed is not critical
- **Mean Flow**: When you need fast generation but can tolerate some training instability
- **Improved Mean Flow**: **Recommended** - Best balance of speed, stability, and quality

---

## References

1. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", 2023
   - https://arxiv.org/pdf/2210.02747

2. **Mean Flow**: Geng et al., "Mean Flows for One-step Generative Modeling", 2025
   - https://arxiv.org/abs/2505.13447

3. **Improved Mean Flow**: Geng et al., "Improved Mean Flows: On the Challenges of Fastforward Generative Models", 2024
   - https://arxiv.org/abs/2512.02012


