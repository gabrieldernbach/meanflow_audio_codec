"""Core MDCT test: implementation correctness verification."""

import jax.numpy as jnp
import numpy as np

from meanflow_audio_codec.preprocessing.mdct import mdct, imdct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from test_mdct_utils import imdct_baseline, mdct_baseline


def test_mdct_baseline_vs_optimized():
    """Test that optimized JAX implementation matches baseline NumPy implementation."""
    np.random.seed(42)
    window_size = 256
    hop_size = window_size // 2
    signal_length = 1024
    
    x_np = np.random.randn(signal_length).astype(np.float32)
    x_jax = jnp.array(x_np)
    
    # Baseline adds batch dimension for 1D input: (T,) -> (1, T) -> (1, n_frames, window_size)
    X_baseline = mdct_baseline(x_np, window_size, hop_size)
    # Optimized preserves shape: (T,) -> (n_frames, window_size)
    X_optimized = mdct(x_jax, window_size, hop_size)
    X_optimized_np = np.array(X_optimized)
    
    # Compare: baseline is (1, n_frames, window_size), optimized is (n_frames, window_size)
    assert X_baseline.shape[1:] == X_optimized_np.shape, \
        f"Shape mismatch: baseline {X_baseline.shape[1:]} vs optimized {X_optimized_np.shape}"
    
    # Compare values (squeeze batch dim from baseline)
    np.testing.assert_allclose(
        X_baseline.squeeze(0), X_optimized_np, rtol=1e-4, atol=1e-3
    )
    
    # Test inverse transform
    x_recon_baseline = imdct_baseline(X_baseline, window_size, hop_size)
    x_recon_optimized = imdct(X_optimized, window_size, hop_size)
    x_recon_optimized_np = np.array(x_recon_optimized)
    
    # Both should have same output length
    min_len = min(
        x_recon_baseline.shape[-1],
        x_recon_optimized_np.shape[-1],
        signal_length,
    )
    
    # Compare reconstruction (squeeze batch dim from baseline)
    np.testing.assert_allclose(
        x_recon_baseline.squeeze(0)[..., :min_len],
        x_recon_optimized_np[..., :min_len],
        rtol=1e-4,
        atol=1e-3,
    )
