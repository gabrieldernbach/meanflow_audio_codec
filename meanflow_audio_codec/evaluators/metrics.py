import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter


def _sqrtm_psd(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Square root of positive semi-definite matrix via eigendecomposition.
    
    Args:
        matrix: Input matrix, shape [N, N].
        eps: Minimum eigenvalue threshold.
        
    Returns:
        Square root matrix, shape [N, N].
    """
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, eps, None)
    return vecs @ (np.sqrt(vals)[None] * vecs.T)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Fréchet distance between two Gaussian distributions.
    
    Computes FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂)), where μ and Σ are
    the mean and covariance of the two distributions.
    
    Args:
        mu1: Mean of first distribution, shape [D].
        sigma1: Covariance of first distribution, shape [D, D].
        mu2: Mean of second distribution, shape [D].
        sigma2: Covariance of second distribution, shape [D, D].
        
    Returns:
        Fréchet distance (scalar).
    """
    diff = mu1 - mu2
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * 1e-6
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * 1e-6
    sqrt_sigma1 = _sqrtm_psd(sigma1)
    covmean = _sqrtm_psd(sqrt_sigma1 @ sigma2 @ sqrt_sigma1)
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def kid_score(
    emb_real: np.ndarray,
    emb_fake: np.ndarray,
    subset_size: int = 100,
    num_subsets: int = 50,
    seed: int = 0,
) -> float:
    """Kernel Inception Distance (KID) score.
    
    Computes KID via unbiased MMD estimator with polynomial kernel k(x,y) = (x·y/d + 1)³
    over multiple random subsets for stability.
    
    Args:
        emb_real: Real embeddings, shape [N_real, D].
        emb_fake: Fake embeddings, shape [N_fake, D].
        subset_size: Number of samples per subset.
        num_subsets: Number of subsets to average over.
        seed: Random seed for subset sampling.
        
    Returns:
        KID score (scalar).
    """
    rng = np.random.default_rng(seed)
    n_real = emb_real.shape[0]
    n_fake = emb_fake.shape[0]
    subset_size = min(subset_size, n_real, n_fake)
    if subset_size < 2:
        raise ValueError("subset_size must be >= 2 for KID computation")
    dim = emb_real.shape[1]

    def kernel(x, y):
        return ((x @ y.T) / float(dim) + 1.0) ** 3

    mmds = []
    for _ in range(num_subsets):
        idx_real = rng.choice(n_real, subset_size, replace=False)
        idx_fake = rng.choice(n_fake, subset_size, replace=False)
        x = emb_real[idx_real]
        y = emb_fake[idx_fake]

        k_xx = kernel(x, x)
        k_yy = kernel(y, y)
        k_xy = kernel(x, y)

        sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / (subset_size * (subset_size - 1))
        sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / (subset_size * (subset_size - 1))
        sum_xy = np.mean(k_xy)
        mmd = sum_xx + sum_yy - 2.0 * sum_xy
        mmds.append(mmd)

    return float(np.mean(mmds))


def psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float | None = None,
) -> float:
    """Peak Signal-to-Noise Ratio (PSNR).
    
    Computes PSNR = 20 * log10(MAX_VAL / sqrt(MSE)), where MAX_VAL is the
    maximum possible value in the data range.
    
    Args:
        pred: Predicted values, shape [B, ...] or [...]
        target: Target values, shape [B, ...] or [...]
        data_range: Maximum possible value. If None, inferred from target:
            - If target in [-1, 1] range, data_range = 2.0
            - If target in [0, 1] range, data_range = 1.0
            - Otherwise, data_range = target.max() - target.min()
    
    Returns:
        PSNR in dB (scalar).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Compute MSE
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    
    # Infer data range if not provided
    if data_range is None:
        target_min, target_max = target.min(), target.max()
        if target_min >= -1.1 and target_max <= 1.1:
            data_range = 2.0
        elif target_min >= -0.1 and target_max <= 1.1:
            data_range = 1.0
        else:
            data_range = target_max - target_min
    
    if data_range <= 0:
        raise ValueError(f"Invalid data_range: {data_range}")
    
    psnr_val = 20.0 * np.log10(data_range / np.sqrt(mse))
    return float(psnr_val)


def ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float | None = None,
    win_size: int = 11,
    gaussian_weights: bool = True,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM).
    
    Computes SSIM between predicted and target images using a Gaussian window.
    Supports both grayscale and multi-channel images.
    
    Args:
        pred: Predicted values, shape [B, H, W] or [B, H, W, C] or [H, W] or [H, W, C]
        target: Target values, same shape as pred
        data_range: Maximum possible value. If None, inferred from target
        win_size: Size of the Gaussian window (must be odd)
        gaussian_weights: If True, use Gaussian weights; if False, use uniform
        sigma: Standard deviation for Gaussian window
        k1: Constant for luminance term stability
        k2: Constant for contrast term stability
    
    Returns:
        Mean SSIM over batch (scalar).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Handle batched vs unbatched
    if pred.ndim == 2:
        pred = pred[None, ...]
        target = target[None, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Ensure win_size is odd
    if win_size % 2 == 0:
        win_size += 1
    
    # Infer data range if not provided
    if data_range is None:
        target_min, target_max = target.min(), target.max()
        if target_min >= -1.1 and target_max <= 1.1:
            data_range = 2.0
        elif target_min >= -0.1 and target_max <= 1.1:
            data_range = 1.0
        else:
            data_range = target_max - target_min
    
    if data_range <= 0:
        raise ValueError(f"Invalid data_range: {data_range}")
    
    # Create Gaussian window
    if gaussian_weights:
        gaussian_window = gaussian_filter(
            np.ones((win_size, win_size), dtype=np.float64), sigma=sigma
        )
        gaussian_window = gaussian_window / gaussian_window.sum()
    else:
        gaussian_window = np.ones((win_size, win_size), dtype=np.float64) / (win_size ** 2)
    
    # Constants
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    # Compute SSIM for each image in batch
    ssim_values = []
    for i in range(pred.shape[0]):
        img1 = pred[i]
        img2 = target[i]
        
        # Handle multi-channel images
        if img1.ndim == 3:
            # Multi-channel: compute SSIM per channel and average
            channel_ssims = []
            for c in range(img1.shape[2]):
                ch1 = img1[:, :, c]
                ch2 = img2[:, :, c]
                ssim_val = _ssim_single_channel(ch1, ch2, gaussian_window, c1, c2)
                channel_ssims.append(ssim_val)
            ssim_values.append(np.mean(channel_ssims))
        else:
            # Grayscale
            ssim_val = _ssim_single_channel(img1, img2, gaussian_window, c1, c2)
            ssim_values.append(ssim_val)
    
    result = float(np.mean(ssim_values))
    return result if not squeeze_output else result


def _ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    window: np.ndarray,
    c1: float,
    c2: float,
) -> float:
    """Compute SSIM for a single channel."""
    mu1 = signal.convolve2d(img1, window, mode="valid", boundary="symm")
    mu2 = signal.convolve2d(img2, window, mode="valid", boundary="symm")
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = signal.convolve2d(img1 ** 2, window, mode="valid", boundary="symm") - mu1_sq
    sigma2_sq = signal.convolve2d(img2 ** 2, window, mode="valid", boundary="symm") - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, mode="valid", boundary="symm") - mu1_mu2
    
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


