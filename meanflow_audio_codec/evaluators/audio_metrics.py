"""Audio-specific evaluation metrics.

This module provides metrics for evaluating audio reconstruction quality,
including perceptual metrics (PESQ, STOI) and spectral distance metrics.
"""

import numpy as np

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None


def pesq_score(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> float:
    """Perceptual Evaluation of Speech Quality (PESQ) score.
    
    PESQ is a perceptual metric that predicts subjective speech quality.
    Higher scores indicate better quality (range typically -0.5 to 4.5).
    
    Args:
        reference: Reference audio signal, shape [T] or [B, T]
        degraded: Degraded/reconstructed audio signal, same shape as reference
        sample_rate: Sample rate in Hz (must be 8000 or 16000)
        mode: PESQ mode, "wb" (wideband) or "nb" (narrowband)
    
    Returns:
        PESQ score (scalar). Returns NaN if pesq package is not available.
    
    Raises:
        ValueError: If sample_rate is not 8000 or 16000
        ImportError: If pesq package is not installed (only if called)
    """
    if pesq is None:
        raise ImportError(
            "pesq package is required for PESQ computation. "
            "Install with: pip install pesq"
        )
    
    if sample_rate not in (8000, 16000):
        raise ValueError(f"sample_rate must be 8000 or 16000, got {sample_rate}")
    
    reference = np.asarray(reference, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    
    # Handle batched input
    if reference.ndim == 2:
        scores = []
        for i in range(reference.shape[0]):
            score = pesq(sample_rate, reference[i], degraded[i], mode=mode)
            scores.append(score)
        return float(np.mean(scores))
    else:
        score = pesq(sample_rate, reference, degraded, mode=mode)
        return float(score)


def stoi_score(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """Short-Time Objective Intelligibility (STOI) score.
    
    STOI predicts the intelligibility of degraded speech signals.
    Higher scores indicate better intelligibility (range 0.0 to 1.0).
    
    Args:
        reference: Reference audio signal, shape [T] or [B, T]
        degraded: Degraded/reconstructed audio signal, same shape as reference
        sample_rate: Sample rate in Hz
        extended: If True, use extended STOI (eSTOI)
    
    Returns:
        STOI score (scalar). Returns NaN if pystoi package is not available.
    
    Raises:
        ImportError: If pystoi package is not installed (only if called)
    """
    if stoi is None:
        raise ImportError(
            "pystoi package is required for STOI computation. "
            "Install with: pip install pystoi"
        )
    
    reference = np.asarray(reference, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    
    # Handle batched input
    if reference.ndim == 2:
        scores = []
        for i in range(reference.shape[0]):
            score = stoi(reference[i], degraded[i], sample_rate, extended=extended)
            scores.append(score)
        return float(np.mean(scores))
    else:
        score = stoi(reference, degraded, sample_rate, extended=extended)
        return float(score)


def spectral_distance(
    reference: np.ndarray,
    degraded: np.ndarray,
    domain: str = "mdct",
    window_size: int = 512,
    hop_size: int | None = None,
) -> float:
    """Spectral distance in frequency domain.
    
    Computes L2 distance between reference and degraded signals in the
    frequency domain (MDCT or mel-spectrogram).
    
    Args:
        reference: Reference signal, shape [T] or [B, T]
        degraded: Degraded/reconstructed signal, same shape as reference
        domain: Frequency domain, "mdct" or "mel"
        window_size: Window size for MDCT (if domain="mdct")
        hop_size: Hop size for MDCT (if domain="mdct", defaults to window_size // 2)
    
    Returns:
        Mean L2 spectral distance (scalar).
    """
    reference = np.asarray(reference, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    
    if reference.shape != degraded.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs degraded {degraded.shape}"
        )
    
    if domain == "mdct":
        from meanflow_audio_codec.preprocessing.mdct import mdct
        
        if hop_size is None:
            hop_size = window_size // 2
        
        # Compute MDCT for both signals
        if reference.ndim == 2:
            # Batched: compute for each sample
            distances = []
            for i in range(reference.shape[0]):
                ref_mdct = mdct(
                    reference[i : i + 1],
                    window_size=window_size,
                    hop_size=hop_size,
                )
                deg_mdct = mdct(
                    degraded[i : i + 1],
                    window_size=window_size,
                    hop_size=hop_size,
                )
                # Flatten and compute L2 distance
                ref_flat = ref_mdct.flatten()
                deg_flat = deg_mdct.flatten()
                dist = np.sqrt(np.mean((ref_flat - deg_flat) ** 2))
                distances.append(dist)
            return float(np.mean(distances))
        else:
            ref_mdct = mdct(reference[None, :], window_size=window_size, hop_size=hop_size)
            deg_mdct = mdct(degraded[None, :], window_size=window_size, hop_size=hop_size)
            ref_flat = ref_mdct.flatten()
            deg_flat = deg_mdct.flatten()
            dist = np.sqrt(np.mean((ref_flat - deg_flat) ** 2))
            return float(dist)
    
    elif domain == "mel":
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for mel-spectrogram computation. "
                "Install with: pip install librosa"
            )
        
        # Compute mel-spectrograms
        if reference.ndim == 2:
            distances = []
            for i in range(reference.shape[0]):
                ref_mel = librosa.feature.melspectrogram(
                    y=reference[i], sr=16000, n_mels=80
                )
                deg_mel = librosa.feature.melspectrogram(
                    y=degraded[i], sr=16000, n_mels=80
                )
                # Convert to log scale and compute L2 distance
                ref_log = np.log(ref_mel + 1e-10)
                deg_log = np.log(deg_mel + 1e-10)
                dist = np.sqrt(np.mean((ref_log - deg_log) ** 2))
                distances.append(dist)
            return float(np.mean(distances))
        else:
            ref_mel = librosa.feature.melspectrogram(y=reference, sr=16000, n_mels=80)
            deg_mel = librosa.feature.melspectrogram(y=degraded, sr=16000, n_mels=80)
            ref_log = np.log(ref_mel + 1e-10)
            deg_log = np.log(deg_mel + 1e-10)
            dist = np.sqrt(np.mean((ref_log - deg_log) ** 2))
            return float(dist)
    
    else:
        raise ValueError(f"Invalid domain: {domain}. Must be 'mdct' or 'mel'")

