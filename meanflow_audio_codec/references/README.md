# Reference Implementations

This directory contains reference implementations from external repositories for comparison and learning purposes.

## Source Repository

These implementations are copied from the [genai repository](https://github.com/gabrieldernbach/genai) by Gabriel Dernbach.

## Files

### `flow.py`

PyTorch implementation of **conditional Flow Matching** for MNIST digit generation.

- **Method**: Flow Matching (continuous normalizing flows)
- **Conditioning**: Class-conditional generation (MNIST digit classes 0-9)
- **Architecture**: MLP-based residual blocks with adaptive layer normalization
- **Loss**: Flow matching objective (see equation 23 in https://arxiv.org/pdf/2210.02747)
- **Sampling**: Heun's method (2nd order Runge-Kutta) ODE solver

Key components:
- `ConditionalFlow`: Main model with class embeddings and time embeddings
- `ConditionalResidualBlock`: Residual block with feature-wise modulation (AdaLN)
- Training uses flow matching loss with linear interpolation schedule
- Sampling uses Heun's method ODE solver over 100 steps (default)

### `mean_flow.py`

PyTorch implementation of **Mean Flow** for MNIST digit generation.

- **Method**: Mean Flow
- **Conditioning**: Class-conditional generation (MNIST digit classes 0-9)
- **Architecture**: Similar MLP-based residual blocks as flow matching
- **Loss**: Mean flow loss with adaptive reweighting (gamma=0.5, c=1e-3)
- **Sampling**: Heun's method (2nd order Runge-Kutta) ODE solver over few steps (n_steps=2 default)

Key differences from flow matching:
- Uses both time `t` and reference time `r` embeddings (r ≤ t)
- Mean flow loss with adaptive reweighting based on prediction error
- Requires Jacobian-vector product (JVP) for computing time derivative
- More efficient sampling (typically 2-5 steps vs 100 steps)

### `improved_mean_flow.py`

PyTorch implementation of **Improved Mean Flow** for MNIST digit generation.

- **Method**: Improved Mean Flow (iMF)
- **Conditioning**: Class-conditional generation (MNIST digit classes 0-9)
- **Architecture**: Similar MLP-based residual blocks as flow matching
- **Loss**: Improved mean flow loss with v-loss formulation
- **Sampling**: Heun's method (2nd order Runge-Kutta) ODE solver over few steps (n_steps=2 default)

Key differences from mean flow:
- Boundary condition: v_theta(z_t, t) = u_theta(z_t, t, t)
- JVP uses v_theta instead of e - x
- Compound prediction: V_theta = u_theta + (t-r) * sg(JVP)
- Standard L2 loss (no adaptive reweighting)

## Usage Notes

These are reference implementations and may require:
- `torch`
- `einops`
- `matplotlib`
- `tqdm`
- `tensorflow_datasets` (via `meanflow_audio_codec.datasets.mnist`)

The implementations are self-contained training scripts that can be run directly, though they may need adjustments for your specific environment (e.g., device configuration).

**Note**: These files are now part of the `meanflow_audio_codec` package and can be imported using:
```python
from meanflow_audio_codec.references.flow import ConditionalFlow, Config
from meanflow_audio_codec.references.mean_flow import ConditionalFlow, Config
from meanflow_audio_codec.references.improved_mean_flow import ConditionalFlow, Config
```

