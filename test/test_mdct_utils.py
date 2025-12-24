"""Test utilities for MDCT module.

This module contains baseline NumPy implementations of MDCT/IMDCT for testing
purposes. These implementations are slower but serve as reference implementations
to verify the correctness of the optimized JAX implementations.
"""
import numpy as np


def mdct_baseline(x: np.ndarray, window_size: int, hop_size: int | None = None) -> np.ndarray:
    """Baseline NumPy MDCT for testing.
    
    This is a reference implementation using pure NumPy. It is slower than the
    optimized JAX implementation but serves as a correctness check.
    
    Args:
        x: Input signal of shape `(..., T)` where `T` is the time dimension.
        window_size: Size of the MDCT window. Must be positive.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`.
    
    Returns:
        MDCT coefficients of shape `(..., n_frames, window_size)`.
    """
    hop_size = hop_size or window_size // 2
    w2n = np.sin(np.pi * (np.arange(2 * window_size, dtype=np.float32) + 0.5) / (2 * window_size))
    x = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    b, T = x.shape
    nf = 1 if T < window_size else (T - window_size) // hop_size + 1
    need = (nf - 1) * hop_size + 2 * window_size
    if T < need:
        x = np.pad(x, ((0, 0), (0, need - T)), mode="constant")
    
    n, k = np.arange(2 * window_size, dtype=x.dtype)[:, None], np.arange(window_size, dtype=x.dtype)[None, :]
    basis = np.cos(np.pi / window_size * (n + window_size / 2 + 0.5) * (k + 0.5))
    
    coeffs = [np.dot(x[:, i * hop_size:i * hop_size + 2 * window_size] * w2n[None, :], basis) for i in range(nf)]
    return np.stack(coeffs, axis=1).reshape(x.shape[:-1] + (nf, window_size))


def imdct_baseline(X: np.ndarray, window_size: int, hop_size: int | None = None) -> np.ndarray:
    """Baseline NumPy IMDCT for testing.
    
    This is a reference implementation using pure NumPy. It is slower than the
    optimized JAX implementation but serves as a correctness check.
    
    Args:
        X: MDCT coefficients of shape `(..., n_frames, window_size)`.
        window_size: Size of the MDCT window. Must be positive.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`.
    
    Returns:
        Reconstructed signal of shape `(..., T)` where `T` is the time dimension.
    """
    hop_size = hop_size or window_size // 2
    w2n = np.sin(np.pi * (np.arange(2 * window_size, dtype=np.float32) + 0.5) / (2 * window_size))
    X = np.asarray(X, dtype=np.float32).reshape(-1, X.shape[-2], X.shape[-1])
    b, nf = X.shape[:2]
    out_len = (nf - 1) * hop_size + 2 * window_size
    
    n, k = np.arange(2 * window_size, dtype=X.dtype)[:, None], np.arange(window_size, dtype=X.dtype)[None, :]
    basis = np.cos(np.pi / window_size * (n + window_size / 2 + 0.5) * (k + 0.5))
    
    frames = [(2.0 / window_size) * np.dot(X[:, i, :], basis.T) * w2n[None, :] for i in range(nf)]
    
    out = np.zeros((b, out_len), dtype=X.dtype)
    for i in range(nf):
        s, e = i * hop_size, i * hop_size + 2 * window_size
        out[:, s:e] += frames[i] if e <= out_len else frames[i][:, :out_len - s]
    
    return out.reshape(X.shape[:-2] + (out_len,))

