"""MDCT (Modified Discrete Cosine Transform) implementation.

This module provides forward and inverse MDCT transforms for audio processing,
with automatic selection between FFT-based (O(N log N)) and direct (O(N²))
implementations based on window size and backend capabilities.

The MDCT is a lapped transform commonly used in audio codecs. It provides
perfect reconstruction when used with appropriate windowing and overlap-add.

Example:
    >>> import jax.numpy as jnp
    >>> from meanflow_audio_codec.preprocessing.mdct import mdct, imdct, MDCTConfig
    >>> 
    >>> # Basic usage
    >>> x = jnp.random.randn(1, 1000)
    >>> X = mdct(x, window_size=512)
    >>> x_recon = imdct(X, window_size=512)
    >>> 
    >>> # Using configuration object
    >>> config = MDCTConfig(window_size=512, hop_size=256)
    >>> X = mdct(x, config=config)
    >>> x_recon = imdct(X, config=config)

Note:
    On Metal backends (Apple Silicon), the FFT-based implementation is not
    available due to lack of complex64 support. The direct implementation
    is automatically used in this case.
"""
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Constants
DEFAULT_WINDOW_SIZE = 576
DEFAULT_FFT_THRESHOLD = 512
PRINCEN_BRADLEY_OFFSET = 0.5
IMDCT_SCALING_FACTOR = 2.0


@dataclass
class MDCTConfig:
    """Configuration for MDCT transforms.
    
    This configuration object allows you to specify MDCT parameters once and
    reuse them across multiple transform calls, reducing parameter repetition
    in pipelines.
    
    Attributes:
        window_size: Size of the MDCT window. Must be positive. Defaults to 576.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`
            (50% overlap). Must be positive if provided.
        use_fft_threshold: Minimum window size to use FFT-based implementation.
            For smaller windows or Metal backends, the direct implementation is used.
            Defaults to 512.
    
    Example:
        >>> config = MDCTConfig(window_size=512, hop_size=256)
        >>> X = mdct(x, config=config)
        >>> x_recon = imdct(X, config=config)
    """
    window_size: int = DEFAULT_WINDOW_SIZE
    hop_size: int | None = None
    use_fft_threshold: int = DEFAULT_FFT_THRESHOLD
    
    def __post_init__(self) -> None:
        """Validate and set default hop_size."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.hop_size is not None and self.hop_size <= 0:
            raise ValueError(f"hop_size must be positive if provided, got {self.hop_size}")
        if self.use_fft_threshold <= 0:
            raise ValueError(f"use_fft_threshold must be positive, got {self.use_fft_threshold}")
        if self.hop_size is None:
            self.hop_size = self.window_size // 2


# ============================================================================
# Backend detection
# ============================================================================

def _is_metal() -> bool:
    """Check if using Metal backend (complex64 not supported).
    
    Returns:
        True if running on Metal backend, False otherwise.
    """
    try:
        return jax.default_backend().lower() == "metal" or any(d.platform == "metal" for d in jax.devices())
    except Exception:
        return False


# ============================================================================
# Window functions
# ============================================================================

def sine_window(window_length: int) -> jnp.ndarray:
    """Generate sine window w[n] = sin(π(n+0.5)/N) for n in [0, N-1].
    
    This window satisfies the Princen-Bradley condition for perfect reconstruction
    in MDCT transforms.
    
    Args:
        window_length: Window length. Must be positive.
    
    Returns:
        Sine window array of shape `(window_length,)` with dtype `float32`.
    
    Raises:
        ValueError: If window_length is not positive.
    
    Example:
        >>> window = sine_window(512)
        >>> window.shape
        (512,)
    """
    if window_length <= 0:
        raise ValueError(f"Window length must be positive, got {window_length}")
    return jnp.sin(jnp.pi * (jnp.arange(window_length, dtype=jnp.float32) + PRINCEN_BRADLEY_OFFSET) / window_length)


def _window_2n(window_size: int, dtype) -> jnp.ndarray:
    """Generate 2N window satisfying Princen-Bradley condition.
    
    Args:
        window_size: Size of the MDCT window.
        dtype: Data type for the window array.
    
    Returns:
        Window array of shape `(2 * window_size,)`.
    """
    return jnp.sin(jnp.pi * (jnp.arange(2 * window_size, dtype=dtype) + PRINCEN_BRADLEY_OFFSET) / (2 * window_size))


# ============================================================================
# Public API: Forward and inverse MDCT transforms
# ============================================================================

def mdct(
    x: jnp.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int | None = None,
    use_fft_threshold: int = DEFAULT_FFT_THRESHOLD,
    config: MDCTConfig | None = None,
) -> jnp.ndarray:
    """Compute the forward Modified Discrete Cosine Transform (MDCT).
    
    This function automatically selects between FFT-based (O(N log N)) and direct
    (O(N²)) implementations based on the window size and backend capabilities.
    The FFT implementation is faster for larger windows but requires complex64
    support, which is not available on Metal backends.
    
    Args:
        x: Input signal of shape `(..., T)` where `T` is the time dimension.
            Can be batched or unbatched. Must be a JAX array.
        window_size: Size of the MDCT window. Must be positive. Defaults to 576.
            Ignored if `config` is provided.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`
            (50% overlap). Must be positive if provided. Ignored if `config` is provided.
        use_fft_threshold: Minimum window size to use FFT-based implementation.
            For smaller windows or Metal backends, the direct implementation is used.
            Defaults to 512. Ignored if `config` is provided.
        config: Optional configuration object. If provided, overrides individual
            parameters. Defaults to `None`.
    
    Returns:
        MDCT coefficients of shape `(..., n_frames, window_size)` where `n_frames`
        is the number of frames computed from the input signal.
    
    Raises:
        TypeError: If input is not a JAX array.
        ValueError: If window_size is not positive, or if input has invalid shape.
    
    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.random.randn(2, 1000)  # 2 batches, 1000 samples
        >>> X = mdct(x, window_size=512)
        >>> X.shape
        (2, 2, 512)  # 2 batches, 2 frames, 512 coefficients
        
        >>> # Using config object
        >>> config = MDCTConfig(window_size=512, hop_size=256)
        >>> X = mdct(x, config=config)
    """
    if not isinstance(x, jnp.ndarray):
        raise TypeError(f"Input must be a JAX array, got {type(x)}")
    if x.ndim == 0:
        raise ValueError("Input must have at least 1 dimension")
    
    window_size, hop_size, use_fft_threshold = _resolve_config(config, window_size, hop_size, use_fft_threshold)
    
    if _is_metal() or window_size < use_fft_threshold:
        return _mdct_direct(x, window_size, hop_size)
    return mdct_fft(x, window_size, hop_size)


def imdct(
    X: jnp.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int | None = None,
    use_fft_threshold: int = DEFAULT_FFT_THRESHOLD,
    config: MDCTConfig | None = None,
) -> jnp.ndarray:
    """Compute the inverse Modified Discrete Cosine Transform (IMDCT).
    
    This function automatically selects between FFT-based (O(N log N)) and direct
    (O(N²)) implementations based on the window size and backend capabilities.
    The FFT implementation is faster for larger windows but requires complex64
    support, which is not available on Metal backends.
    
    Args:
        X: MDCT coefficients of shape `(..., n_frames, window_size)` where `n_frames`
            is the number of frames. Can be batched or unbatched. Must be a JAX array.
        window_size: Size of the MDCT window. Must be positive. Defaults to 576.
            Ignored if `config` is provided.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`
            (50% overlap). Must be positive if provided. Ignored if `config` is provided.
        use_fft_threshold: Minimum window size to use FFT-based implementation.
            For smaller windows or Metal backends, the direct implementation is used.
            Defaults to 512. Ignored if `config` is provided.
        config: Optional configuration object. If provided, overrides individual
            parameters. Defaults to `None`.
    
    Returns:
        Reconstructed signal of shape `(..., T)` where `T` is the time dimension.
        The output length is `(n_frames - 1) * hop_size + 2 * window_size`.
    
    Raises:
        TypeError: If input is not a JAX array.
        ValueError: If window_size is not positive, or if input has invalid shape.
    
    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.random.randn(2, 3, 512)  # 2 batches, 3 frames, 512 coefficients
        >>> x = imdct(X, window_size=512)
        >>> x.shape
        (2, 1536)  # Reconstructed signal length
        
        >>> # Using config object
        >>> config = MDCTConfig(window_size=512, hop_size=256)
        >>> x = imdct(X, config=config)
    """
    if not isinstance(X, jnp.ndarray):
        raise TypeError(f"Input must be a JAX array, got {type(X)}")
    if X.ndim < 2:
        raise ValueError(f"Input must have at least 2 dimensions (n_frames, window_size), got shape {X.shape}")
    
    window_size, hop_size, use_fft_threshold = _resolve_config(config, window_size, hop_size, use_fft_threshold)
    
    if _is_metal() or window_size < use_fft_threshold:
        return _imdct_direct(X, window_size, hop_size)
    return imdct_fft(X, window_size, hop_size)


# ============================================================================
# Implementation variants: FFT-based (public) and direct (internal)
# ============================================================================

def mdct_fft(x: jnp.ndarray, window_size: int, hop_size: int | None = None) -> jnp.ndarray:
    """FFT-based MDCT implementation.
    
    This is the faster O(N log N) implementation, but requires complex64 support.
    Automatically used by `mdct()` for window sizes >= `use_fft_threshold` on
    non-Metal backends.
    
    Args:
        x: Input signal of shape `(..., T)`.
        window_size: Size of the MDCT window.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`.
    
    Returns:
        MDCT coefficients of shape `(..., n_frames, window_size)`.
    """
    window_2n = _window_2n(window_size, x.dtype)
    k = np.arange(window_size, dtype=np.float32)
    twiddle = jnp.asarray(np.exp(-1j * np.pi * (k + PRINCEN_BRADLEY_OFFSET) / window_size).astype(np.complex64))
    x_flat, batch_size, num_frames, hop, original_shape = _prepare_mdct(x, window_size, hop_size)
    
    def frame(i: int) -> jnp.ndarray:
        frame_slice = jax.lax.dynamic_slice(x_flat, (0, i * hop), (batch_size, 2 * window_size))
        return _mdct_fft_frame(frame_slice * window_2n[None, :], window_size, twiddle)
    
    return jax.vmap(frame, in_axes=(0,), out_axes=1)(jnp.arange(num_frames)).reshape(original_shape[:-1] + (num_frames, window_size))


def imdct_fft(X: jnp.ndarray, window_size: int, hop_size: int | None = None) -> jnp.ndarray:
    """FFT-based IMDCT implementation.
    
    This is the faster O(N log N) implementation, but requires complex64 support.
    Automatically used by `imdct()` for window sizes >= `use_fft_threshold` on
    non-Metal backends.
    
    Args:
        X: MDCT coefficients of shape `(..., n_frames, window_size)`.
        window_size: Size of the MDCT window.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`.
    
    Returns:
        Reconstructed signal of shape `(..., T)`.
    """
    window_2n = _window_2n(window_size, X.dtype)
    k = np.arange(window_size, dtype=np.float32)
    twiddle = jnp.asarray(np.exp(1j * np.pi * (k + PRINCEN_BRADLEY_OFFSET) / window_size).astype(np.complex64))
    X_flat, batch_size, num_frames, output_length, original_shape, hop = _prepare_imdct(X, window_size, hop_size)
    
    def frame(i: int) -> jnp.ndarray:
        return _imdct_fft_frame(X_flat[:, i, :], window_size, twiddle) * window_2n[None, :]
    
    frames = jax.vmap(frame, in_axes=(0,), out_axes=0)(jnp.arange(num_frames))
    return _overlap_add(frames, batch_size, num_frames, window_size, hop, output_length).reshape(original_shape[:-2] + (output_length,))


def _mdct_direct(x: jnp.ndarray, window_size: int, hop_size: int | None = None) -> jnp.ndarray:
    """Direct cosine MDCT implementation."""
    window_2n = _window_2n(window_size, x.dtype)
    x_flat, batch_size, num_frames, hop, original_shape = _prepare_mdct(x, window_size, hop_size)
    basis = _cosine_basis(window_size, x.dtype)
    
    def frame(i: int) -> jnp.ndarray:
        frame_slice = jax.lax.dynamic_slice(x_flat, (0, i * hop), (batch_size, 2 * window_size))
        return _mdct_direct_frame(frame_slice * window_2n[None, :], window_size, basis)
    
    return jax.vmap(frame, in_axes=(0,), out_axes=1)(jnp.arange(num_frames)).reshape(original_shape[:-1] + (num_frames, window_size))


def _imdct_direct(X: jnp.ndarray, window_size: int, hop_size: int | None = None) -> jnp.ndarray:
    """Direct cosine IMDCT implementation."""
    window_2n = _window_2n(window_size, X.dtype)
    X_flat, batch_size, num_frames, output_length, original_shape, hop = _prepare_imdct(X, window_size, hop_size)
    basis = _cosine_basis(window_size, X.dtype)
    
    def frame(i: int) -> jnp.ndarray:
        return _imdct_direct_frame(X_flat[:, i, :], window_size, basis) * window_2n[None, :]
    
    frames = jax.vmap(frame, in_axes=(0,), out_axes=0)(jnp.arange(num_frames))
    return _overlap_add(frames, batch_size, num_frames, window_size, hop, output_length).reshape(original_shape[:-2] + (output_length,))


# ============================================================================
# Internal helper functions: Frame processing
# ============================================================================

def _mdct_direct_frame(windowed_frame: jnp.ndarray, window_size: int, basis: jnp.ndarray) -> jnp.ndarray:
    """MDCT single frame via direct cosine computation.
    
    Args:
        windowed_frame: Windowed frame of shape `(batch, 2 * window_size)`.
        window_size: Size of the MDCT window.
        basis: Cosine basis matrix of shape `(2 * window_size, window_size)`.
    
    Returns:
        MDCT coefficients of shape `(batch, window_size)`.
    """
    return jnp.einsum('bi,ij->bj', windowed_frame, basis)


def _imdct_direct_frame(coefficients: jnp.ndarray, window_size: int, basis: jnp.ndarray) -> jnp.ndarray:
    """IMDCT single frame via direct cosine computation.
    
    Args:
        coefficients: MDCT coefficients of shape `(batch, window_size)`.
        window_size: Size of the MDCT window.
        basis: Cosine basis matrix of shape `(2 * window_size, window_size)`.
    
    Returns:
        Reconstructed frame of shape `(batch, 2 * window_size)`.
    """
    return (IMDCT_SCALING_FACTOR / window_size) * jnp.einsum('bi,ij->bj', coefficients, basis.T)


def _mdct_fft_frame(windowed_frame: jnp.ndarray, window_size: int, twiddle: jnp.ndarray) -> jnp.ndarray:
    """MDCT single frame via FFT.
    
    Args:
        windowed_frame: Windowed frame of shape `(batch, 2 * window_size)`.
        window_size: Size of the MDCT window.
        twiddle: Twiddle factors for FFT of shape `(window_size,)`.
    
    Returns:
        MDCT coefficients of shape `(batch, window_size)`.
    """
    y = -windowed_frame[:, 2*window_size-1:window_size-1:-1] - windowed_frame[:, :window_size]
    return jnp.real(jnp.fft.fft(_to_complex64(y), n=window_size) * twiddle[None, :])


def _imdct_fft_frame(coefficients: jnp.ndarray, window_size: int, twiddle: jnp.ndarray) -> jnp.ndarray:
    """IMDCT single frame via FFT.
    
    Args:
        coefficients: MDCT coefficients of shape `(batch, window_size)`.
        window_size: Size of the MDCT window.
        twiddle: Twiddle factors for IFFT of shape `(window_size,)`.
    
    Returns:
        Reconstructed frame of shape `(batch, 2 * window_size)`.
    """
    y = jnp.fft.ifft(_to_complex64(coefficients) * twiddle[None, :], n=window_size)
    y_real = jnp.real(y)
    return jnp.concatenate([-y_real[:, window_size-1::-1], y_real], axis=-1)


# ============================================================================
# Internal helper functions: Basis computation and utilities
# ============================================================================

def _cosine_basis(window_size: int, dtype) -> jnp.ndarray:
    """Compute MDCT/IMDCT cosine basis matrix.
    
    Args:
        window_size: Size of the MDCT window.
        dtype: Data type for the basis matrix.
    
    Returns:
        Cosine basis matrix of shape `(2 * window_size, window_size)`.
    """
    n = jnp.arange(2 * window_size, dtype=dtype)[:, None]
    k = jnp.arange(window_size, dtype=dtype)[None, :]
    return jnp.cos(jnp.pi / window_size * (n + window_size / 2 + PRINCEN_BRADLEY_OFFSET) * (k + PRINCEN_BRADLEY_OFFSET))


def _to_complex64(x: jnp.ndarray) -> jnp.ndarray:
    """Convert array to complex64 for FFT operations.
    
    Args:
        x: Input array.
    
    Returns:
        Array converted to complex64.
    """
    return x.astype(jnp.float32) * jnp.complex64(1.0)


def _resolve_config(
    config: MDCTConfig | None,
    window_size: int,
    hop_size: int | None,
    use_fft_threshold: int,
) -> Tuple[int, int, int]:
    """Resolve configuration from config object or individual parameters.
    
    Args:
        config: Optional configuration object.
        window_size: Window size parameter (used if config is None).
        hop_size: Hop size parameter (used if config is None).
        use_fft_threshold: FFT threshold parameter (used if config is None).
    
    Returns:
        Tuple of (window_size, hop_size, use_fft_threshold).
    
    Raises:
        ValueError: If any parameter is invalid.
    """
    if config is not None:
        return config.window_size, config.hop_size, config.use_fft_threshold
    
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if hop_size is not None and hop_size <= 0:
        raise ValueError(f"hop_size must be positive if provided, got {hop_size}")
    if use_fft_threshold <= 0:
        raise ValueError(f"use_fft_threshold must be positive, got {use_fft_threshold}")
    if hop_size is None:
        hop_size = window_size // 2
    
    return window_size, hop_size, use_fft_threshold


# ============================================================================
# Internal helper functions: Input preparation and overlap-add
# ============================================================================

def _prepare_mdct(x: jnp.ndarray, window_size: int, hop_size: int | None) -> Tuple[jnp.ndarray, int, int, int, Tuple[int, ...]]:
    """Prepare input and compute frames for MDCT.
    
    Args:
        x: Input signal of shape `(..., T)`.
        window_size: Size of the MDCT window.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`.
    
    Returns:
        Tuple of (flattened_input, batch_size, num_frames, hop_size, original_shape).
    """
    hop_size = hop_size or window_size // 2
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])
    batch_size, time_length = x_flat.shape
    num_frames = 1 if time_length < window_size else (time_length - window_size) // hop_size + 1
    required_length = (num_frames - 1) * hop_size + 2 * window_size
    if time_length < required_length:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, required_length - time_length)), mode="constant")
    return x_flat, batch_size, num_frames, hop_size, original_shape


def _prepare_imdct(X: jnp.ndarray, window_size: int, hop_size: int | None) -> Tuple[jnp.ndarray, int, int, int, Tuple[int, ...], int]:
    """Prepare input for IMDCT.
    
    Args:
        X: MDCT coefficients of shape `(..., n_frames, window_size)`.
        window_size: Size of the MDCT window.
        hop_size: Hop size between windows. If `None`, defaults to `window_size // 2`.
    
    Returns:
        Tuple of (flattened_coefficients, batch_size, num_frames, output_length, original_shape, hop_size).
    """
    hop_size = hop_size or window_size // 2
    original_shape = X.shape
    X_flat = X.reshape(-1, original_shape[-2], original_shape[-1])
    batch_size, num_frames = X_flat.shape[:2]
    output_length = (num_frames - 1) * hop_size + 2 * window_size
    return X_flat, batch_size, num_frames, output_length, original_shape, hop_size


def _overlap_add(frames: jnp.ndarray, batch_size: int, num_frames: int, window_size: int, hop_size: int, output_length: int) -> jnp.ndarray:
    """Overlap-add frames into output signal.
    
    Args:
        frames: Frames of shape `(num_frames, batch_size, 2 * window_size)`.
        batch_size: Batch size.
        num_frames: Number of frames.
        window_size: Size of the MDCT window.
        hop_size: Hop size between windows.
        output_length: Desired output length.
    
    Returns:
        Reconstructed signal of shape `(batch_size, output_length)`.
    """
    output = jnp.zeros((batch_size, output_length + 2 * window_size), dtype=frames.dtype)
    
    def scan_fn(carry: jnp.ndarray, frame_idx: int) -> Tuple[jnp.ndarray, None]:
        start_idx = frame_idx * hop_size
        frame_slice = jax.lax.dynamic_slice(frames, (frame_idx, 0, 0), (1, batch_size, 2 * window_size))[0]
        output_slice = jax.lax.dynamic_slice(carry, (0, start_idx), (batch_size, 2 * window_size))
        return jax.lax.dynamic_update_slice(carry, output_slice + frame_slice, (0, start_idx)), None
    
    output, _ = jax.lax.scan(scan_fn, output, jnp.arange(num_frames))
    return output[:, :output_length]


# ============================================================================
# Flax Linen layers for neural networks
# ============================================================================

class MDCTLayer(nn.Module):
    """Flax layer for forward MDCT transform.

    This layer can be used as the first layer in a convolutional neural
    network to convert time-domain audio signals to frequency-domain MDCT
    coefficients.

    The layer is stateless (no learnable parameters) and wraps the
    `mdct()` function.

    Attributes:
        window_size: Size of the MDCT window. Must be positive.
            Defaults to 576.
        hop_size: Hop size between windows. If `None`, defaults to
            `window_size // 2` (50% overlap). Must be positive if provided.
        use_fft_threshold: Minimum window size to use FFT-based
            implementation. For smaller windows or Metal backends, the
            direct implementation is used. Defaults to 512.
        config: Optional configuration object. If provided, overrides
            individual parameters. Defaults to `None`.

    Example:
        >>> import jax.numpy as jnp
        >>> from meanflow_audio_codec.preprocessing.mdct import MDCTLayer
        >>>
        >>> layer = MDCTLayer(window_size=512, hop_size=256)
        >>>
        >>> # Mono audio: [B, T] -> [B, n_frames, window_size]
        >>> x_mono = jnp.random.randn(32, 1000)
        >>> X_mono = layer.apply({}, x_mono)  # [32, n_frames, 512]
        >>>
        >>> # Stereo audio: [B, T, 2] -> [B, n_frames, window_size*2]
        >>> x_stereo = jnp.random.randn(32, 1000, 2)
        >>> X_stereo = layer.apply({}, x_stereo)  # [32, n_frames, 1024]
    """
    window_size: int = DEFAULT_WINDOW_SIZE
    hop_size: int | None = None
    use_fft_threshold: int = DEFAULT_FFT_THRESHOLD
    config: MDCTConfig | None = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply MDCT transform to input.

        Args:
            x: Input signal of shape:
                - `[B, T]` for mono audio
                - `[B, T, 2]` for stereo audio (left and right channels)

        Returns:
            MDCT coefficients of shape:
                - `[B, n_frames, window_size]` for mono input
                - `[B, n_frames, window_size * 2]` for stereo input
                (left and right channels concatenated along frequency dim)
        """
        is_stereo = x.ndim == 3 and x.shape[-1] == 2

        if is_stereo:
            if self.config is not None:
                left = mdct(x[:, :, 0], config=self.config)
                right = mdct(x[:, :, 1], config=self.config)
            else:
                left = mdct(x[:, :, 0], window_size=self.window_size, hop_size=self.hop_size, use_fft_threshold=self.use_fft_threshold)
                right = mdct(x[:, :, 1], window_size=self.window_size, hop_size=self.hop_size, use_fft_threshold=self.use_fft_threshold)
            return jnp.concatenate([left, right], axis=-1)
        else:
            if self.config is not None:
                return mdct(x, config=self.config)
            return mdct(x, window_size=self.window_size, hop_size=self.hop_size, use_fft_threshold=self.use_fft_threshold)


class IMDCTLayer(nn.Module):
    """Flax layer for inverse MDCT transform.

    This layer can be used as the last layer in a convolutional neural
    network to convert frequency-domain MDCT coefficients back to
    time-domain audio signals.

    The layer is stateless (no learnable parameters) and wraps the
    `imdct()` function.

    Attributes:
        window_size: Size of the MDCT window. Must be positive.
            Defaults to 576.
        hop_size: Hop size between windows. If `None`, defaults to
            `window_size // 2` (50% overlap). Must be positive if provided.
        use_fft_threshold: Minimum window size to use FFT-based
            implementation. For smaller windows or Metal backends, the
            direct implementation is used. Defaults to 512.
        config: Optional configuration object. If provided, overrides
            individual parameters. Defaults to `None`.

    Example:
        >>> import jax.numpy as jnp
        >>> from meanflow_audio_codec.preprocessing.mdct import IMDCTLayer
        >>>
        >>> layer = IMDCTLayer(window_size=512, hop_size=256)
        >>>
        >>> # Mono: [B, n_frames, window_size] -> [B, T]
        >>> X_mono = jnp.random.randn(32, 10, 512)
        >>> x_mono = layer.apply({}, X_mono)  # [32, T]
        >>>
        >>> # Stereo: [B, n_frames, window_size*2] -> [B, T, 2]
        >>> X_stereo = jnp.random.randn(32, 10, 1024)
        >>> x_stereo = layer.apply({}, X_stereo)  # [32, T, 2]
    """
    window_size: int = DEFAULT_WINDOW_SIZE
    hop_size: int | None = None
    use_fft_threshold: int = DEFAULT_FFT_THRESHOLD
    config: MDCTConfig | None = None
    
    @nn.compact
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Apply inverse MDCT transform to input.

        Args:
            X: MDCT coefficients of shape:
                - `[B, n_frames, window_size]` for mono
                - `[B, n_frames, window_size * 2]` for stereo
                (concatenated left and right channels)

        Returns:
            Reconstructed signal of shape:
                - `[B, T]` for mono output
                - `[B, T, 2]` for stereo output
            where `T` is the time dimension with length
            `(n_frames - 1) * hop_size + 2 * window_size`.
        """
        window_size = self.window_size if self.config is None else self.config.window_size
        is_stereo = X.shape[-1] == window_size * 2

        if is_stereo:
            left_coeffs = X[:, :, :window_size]
            right_coeffs = X[:, :, window_size:]

            if self.config is not None:
                left = imdct(left_coeffs, config=self.config)
                right = imdct(right_coeffs, config=self.config)
            else:
                left = imdct(left_coeffs, window_size=self.window_size, hop_size=self.hop_size, use_fft_threshold=self.use_fft_threshold)
                right = imdct(right_coeffs, window_size=self.window_size, hop_size=self.hop_size, use_fft_threshold=self.use_fft_threshold)

            return jnp.stack([left, right], axis=-1)
        else:
            if self.config is not None:
                return imdct(X, config=self.config)
            return imdct(X, window_size=self.window_size, hop_size=self.hop_size, use_fft_threshold=self.use_fft_threshold)
