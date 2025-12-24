# Reference Implementations

This directory contains reference implementations from external repositories for comparison and learning purposes.

## Source Repository

These implementations are copied from the [genai repository](https://github.com/gabrieldernbach/genai) by Gabriel Dernbach.

## Files

### `flow_matching_mnist.py`

PyTorch implementation of **conditional Flow Matching** for MNIST digit generation.

- **Method**: Flow Matching (continuous normalizing flows)
- **Conditioning**: Class-conditional generation (MNIST digit classes 0-9)
- **Architecture**: MLP-based residual blocks with adaptive layer normalization
- **Loss**: Flow matching objective (see equation 23 in https://arxiv.org/pdf/2210.02747)
- **Sampling**: Runge-Kutta 4th order ODE solver

Key components:
- `ConditionalFlow`: Main model with class embeddings and time embeddings
- `ConditionalResidualBlock`: Residual block with feature-wise modulation (AdaLN)
- Training uses flow matching loss with linear interpolation schedule
- Sampling uses RK4 ODE solver over 100 steps (default)

### `mean_flow_mnist.py`

PyTorch implementation of **Mean Flow** for MNIST digit generation.

- **Method**: Mean Flow (Improved Mean Flow method)
- **Conditioning**: Class-conditional generation (MNIST digit classes 0-9)
- **Architecture**: Similar MLP-based residual blocks as flow matching
- **Loss**: Mean flow loss with adaptive reweighting (gamma=0.5, c=1e-3)
- **Sampling**: Simple Euler-style ODE solver over very few steps (n_steps=5 default)

Key differences from flow matching:
- Uses both time `t` and reference time `r` embeddings (r ≤ t)
- Mean flow loss with adaptive reweighting based on prediction error
- Requires Jacobian-vector product (JVP) for computing time derivative
- More efficient sampling (typically 2-5 steps vs 100 steps)

**⚠️ Note**: This implementation has been corrected based on comparison with official implementations. See `mean_flow_mnist_ISSUES.md` for details about the fixes applied (JVP call and time clipping).

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
from meanflow_audio_codec.references.mean_flow_mnist import ConditionalFlow, Config
```

