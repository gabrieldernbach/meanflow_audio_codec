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
    
    def __post_init__(self):
        """Validate and set default hop_size."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.hop_size is not None and self.hop_size <= 0:
            raise ValueError(f"hop_size must be positive if provided, got {self.hop_size}")
        if self.use_fft_threshold <= 0:
            raise ValueError(f"use_fft_threshold must be positive, got {self.use_fft_threshold}")
        if self.hop_size is None:
            self.hop_size = self.window_size // 2


def _is_metal() -> bool:
    """Check if using Metal backend (complex64 not supported).
    
    Returns:
        True if running on Metal backend, False otherwise.
    """
    try:
        return jax.default_backend().lower() == "metal" or any(d.platform == "metal" for d in jax.devices())
    except Exception:
        return False


def sine_window(n: int) -> jnp.ndarray:
    """Generate sine window w[n] = sin(π(n+0.5)/N) for n in [0, N-1].
    
    This window satisfies the Princen-Bradley condition for perfect reconstruction
    in MDCT transforms.
    
    Args:
        n: Window length. Must be positive.
    
    Returns:
        Sine window array of shape `(n,)` with dtype `float32`.
    
    Example:
        >>> window = sine_window(512)
        >>> window.shape
        (512,)
    """
    if n <= 0:
        raise ValueError(f"Window length must be positive, got {n}")
    return jnp.sin(jnp.pi * (jnp.arange(n, dtype=jnp.float32) + PRINCEN_BRADLEY_OFFSET) / n)


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
    # Validate input
    if not isinstance(x, jnp.ndarray):
        raise TypeError(f"Input must be a JAX array, got {type(x)}")
    if x.ndim == 0:
        raise ValueError("Input must have at least 1 dimension")
    
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        window_size = config.window_size
        hop_size = config.hop_size
        use_fft_threshold = config.use_fft_threshold
    else:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if hop_size is not None and hop_size <= 0:
            raise ValueError(f"hop_size must be positive if provided, got {hop_size}")
        if use_fft_threshold <= 0:
            raise ValueError(f"use_fft_threshold must be positive, got {use_fft_threshold}")
        if hop_size is None:
            hop_size = window_size // 2
    
    # Select implementation based on backend and window size
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
    # Validate input
    if not isinstance(X, jnp.ndarray):
        raise TypeError(f"Input must be a JAX array, got {type(X)}")
    if X.ndim < 2:
        raise ValueError(f"Input must have at least 2 dimensions (n_frames, window_size), got shape {X.shape}")
    
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        window_size = config.window_size
        hop_size = config.hop_size
        use_fft_threshold = config.use_fft_threshold
    else:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if hop_size is not None and hop_size <= 0:
            raise ValueError(f"hop_size must be positive if provided, got {hop_size}")
        if use_fft_threshold <= 0:
            raise ValueError(f"use_fft_threshold must be positive, got {use_fft_threshold}")
        if hop_size is None:
            hop_size = window_size // 2
    
    # Select implementation based on backend and window size
    if _is_metal() or window_size < use_fft_threshold:
        return _imdct_direct(X, window_size, hop_size)
    return imdct_fft(X, window_size, hop_size)


# Internal helper functions

def _w2n(N: int, dtype) -> jnp.ndarray:
    """Generate 2N window satisfying Princen-Bradley condition."""
    return jnp.sin(jnp.pi * (jnp.arange(2 * N, dtype=dtype) + PRINCEN_BRADLEY_OFFSET) / (2 * N))


def _cos_basis(N: int, dtype) -> jnp.ndarray:
    """MDCT/IMDCT cosine basis matrix."""
    n, k = jnp.arange(2 * N, dtype=dtype)[:, None], jnp.arange(N, dtype=dtype)[None, :]
    return jnp.cos(jnp.pi / N * (n + N / 2 + PRINCEN_BRADLEY_OFFSET) * (k + PRINCEN_BRADLEY_OFFSET))


def _c64(x: jnp.ndarray) -> jnp.ndarray:
    """Convert to complex64 for FFT."""
    return x.astype(jnp.float32) * jnp.complex64(1.0)


def _mdct_direct_frame(wf: jnp.ndarray, N: int, basis: jnp.ndarray) -> jnp.ndarray:
    """MDCT single frame via direct cosine."""
    return jnp.einsum('bi,ij->bj', wf, basis)


def _imdct_direct_frame(Xf: jnp.ndarray, N: int, basis: jnp.ndarray) -> jnp.ndarray:
    """IMDCT single frame via direct cosine."""
    return (IMDCT_SCALING_FACTOR / N) * jnp.einsum('bi,ij->bj', Xf, basis.T)


def _mdct_fft_frame(wf: jnp.ndarray, N: int, tw: jnp.ndarray) -> jnp.ndarray:
    """MDCT single frame via FFT."""
    y = -wf[:, 2*N-1:N-1:-1] - wf[:, :N]
    return jnp.real(jnp.fft.fft(_c64(y), n=N) * tw[None, :])


def _imdct_fft_frame(Xf: jnp.ndarray, N: int, tw: jnp.ndarray) -> jnp.ndarray:
    """IMDCT single frame via FFT."""
    y = jnp.fft.ifft(_c64(Xf) * tw[None, :], n=N)
    yr = jnp.real(y)
    return jnp.concatenate([-yr[:, N-1::-1], yr], axis=-1)


def _prepare_mdct(x: jnp.ndarray, N: int, hop: int | None) -> tuple:
    """Prepare input and compute frames for MDCT."""
    hop = hop or N // 2
    s = x.shape
    x = x.reshape(-1, s[-1])
    b, T = x.shape
    nf = 1 if T < N else (T - N) // hop + 1
    need = (nf - 1) * hop + 2 * N
    if T < need:
        x = jnp.pad(x, ((0, 0), (0, need - T)), mode="constant")
    return x, b, nf, hop, s


def _prepare_imdct(X: jnp.ndarray, N: int, hop: int | None) -> tuple:
    """Prepare input for IMDCT."""
    hop = hop or N // 2
    s = X.shape
    X = X.reshape(-1, s[-2], s[-1])
    b, nf = X.shape[:2]
    out_len = (nf - 1) * hop + 2 * N
    return X, b, nf, out_len, s, hop


def _overlap_add(frames: jnp.ndarray, b: int, nf: int, N: int, hop: int, out_len: int) -> jnp.ndarray:
    """Overlap-add frames into output."""
    out = jnp.zeros((b, out_len + 2 * N), dtype=frames.dtype)
    
    def scan_fn(c, i):
        s = i * hop
        f = jax.lax.dynamic_slice(frames, (i, 0, 0), (1, b, 2 * N))[0]
        sl = jax.lax.dynamic_slice(c, (0, s), (b, 2 * N))
        return jax.lax.dynamic_update_slice(c, sl + f, (0, s)), None
    
    out, _ = jax.lax.scan(scan_fn, out, jnp.arange(nf))
    return out[:, :out_len]


# Implementation variants

def _mdct_direct(x: jnp.ndarray, window_size: int, hop_size: int | None = None) -> jnp.ndarray:
    """Direct cosine MDCT implementation."""
    w2n = _w2n(window_size, x.dtype)
    x, b, nf, hop, s = _prepare_mdct(x, window_size, hop_size)
    basis = _cos_basis(window_size, x.dtype)
    
    def frame(i):
        f = jax.lax.dynamic_slice(x, (0, i * hop), (b, 2 * window_size))
        return _mdct_direct_frame(f * w2n[None, :], window_size, basis)
    
    return jax.vmap(frame, in_axes=(0,), out_axes=1)(jnp.arange(nf)).reshape(s[:-1] + (nf, window_size))


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
    w2n = _w2n(window_size, x.dtype)
    k = np.arange(window_size, dtype=np.float32)
    tw = jnp.asarray(np.exp(-1j * np.pi * (k + PRINCEN_BRADLEY_OFFSET) / window_size).astype(np.complex64))
    x, b, nf, hop, s = _prepare_mdct(x, window_size, hop_size)
    
    def frame(i):
        f = jax.lax.dynamic_slice(x, (0, i * hop), (b, 2 * window_size))
        return _mdct_fft_frame(f * w2n[None, :], window_size, tw)
    
    return jax.vmap(frame, in_axes=(0,), out_axes=1)(jnp.arange(nf)).reshape(s[:-1] + (nf, window_size))


def _imdct_direct(X: jnp.ndarray, window_size: int, hop_size: int | None = None) -> jnp.ndarray:
    """Direct cosine IMDCT implementation."""
    w2n = _w2n(window_size, X.dtype)
    X, b, nf, out_len, s, hop = _prepare_imdct(X, window_size, hop_size)
    basis = _cos_basis(window_size, X.dtype)
    
    def frame(i):
        return _imdct_direct_frame(X[:, i, :], window_size, basis) * w2n[None, :]
    
    frames = jax.vmap(frame, in_axes=(0,), out_axes=0)(jnp.arange(nf))
    return _overlap_add(frames, b, nf, window_size, hop, out_len).reshape(s[:-2] + (out_len,))


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
    w2n = _w2n(window_size, X.dtype)
    k = np.arange(window_size, dtype=np.float32)
    tw = jnp.asarray(np.exp(1j * np.pi * (k + PRINCEN_BRADLEY_OFFSET) / window_size).astype(np.complex64))
    X, b, nf, out_len, s, hop = _prepare_imdct(X, window_size, hop_size)
    
    def frame(i):
        return _imdct_fft_frame(X[:, i, :], window_size, tw) * w2n[None, :]
    
    frames = jax.vmap(frame, in_axes=(0,), out_axes=0)(jnp.arange(nf))
    return _overlap_add(frames, b, nf, window_size, hop, out_len).reshape(s[:-2] + (out_len,))


# Flax Linen layers for neural networks


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
        # Check if stereo (has channel dimension as last dim)
        is_stereo = x.ndim == 3 and x.shape[-1] == 2

        if is_stereo:
            # Process left and right channels independently
            if self.config is not None:
                left = mdct(x[:, :, 0], config=self.config)
                right = mdct(x[:, :, 1], config=self.config)
            else:
                left = mdct(
                    x[:, :, 0],
                    window_size=self.window_size,
                    hop_size=self.hop_size,
                    use_fft_threshold=self.use_fft_threshold,
                )
                right = mdct(
                    x[:, :, 1],
                    window_size=self.window_size,
                    hop_size=self.hop_size,
                    use_fft_threshold=self.use_fft_threshold,
                )
            # Concatenate along frequency dimension:
            # [B, n_frames, window_size*2]
            return jnp.concatenate([left, right], axis=-1)
        else:
            # Mono: process as-is
            if self.config is not None:
                return mdct(x, config=self.config)
            return mdct(
                x,
                window_size=self.window_size,
                hop_size=self.hop_size,
                use_fft_threshold=self.use_fft_threshold,
            )


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
        # Get window_size first
        window_size = (
            self.window_size if self.config is None else self.config.window_size
        )
        # Determine if stereo based on frequency dimension size.
        # Stereo should have window_size * 2 in the last dimension.
        is_stereo = X.shape[-1] == window_size * 2

        if is_stereo:
            # Split left and right channels
            left_coeffs = X[:, :, :window_size]
            right_coeffs = X[:, :, window_size:]

            # Process each channel independently
            if self.config is not None:
                left = imdct(left_coeffs, config=self.config)
                right = imdct(right_coeffs, config=self.config)
            else:
                left = imdct(
                    left_coeffs,
                    window_size=self.window_size,
                    hop_size=self.hop_size,
                    use_fft_threshold=self.use_fft_threshold,
                )
                right = imdct(
                    right_coeffs,
                    window_size=self.window_size,
                    hop_size=self.hop_size,
                    use_fft_threshold=self.use_fft_threshold,
                )

            # Stack along channel dimension: [B, T, 2]
            return jnp.stack([left, right], axis=-1)
        else:
            # Mono: process as-is
            if self.config is not None:
                return imdct(X, config=self.config)
            return imdct(
                X,
                window_size=self.window_size,
                hop_size=self.hop_size,
                use_fft_threshold=self.use_fft_threshold,
            )
