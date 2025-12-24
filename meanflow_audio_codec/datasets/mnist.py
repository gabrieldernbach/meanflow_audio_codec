from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow_datasets as tfds
from toolz import compose

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _preprocess_mnist_images(
    images: np.ndarray,
    format: str = "1d",
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess MNIST images (internal helper).
    
    Args:
        images: Input images of shape [B, H, W] or [B, H*W]
        format: Output format, either "1d" (flattened) or "2d"
        normalize: If True, normalize to [-1, 1] range
    
    Returns:
        Preprocessed images
    """
    if format not in ("1d", "2d"):
        raise ValueError(f"Invalid format: {format}. Must be '1d' or '2d'")

    pipeline = compose(
        lambda x: x.reshape(x.shape[0], -1) if format == "1d" else x,
        lambda x: (x - 0.5) / 0.5 if normalize else x,
        lambda x: x.astype(np.float32) / 255.0,
    )
    return pipeline(images)


# ============================================================================
# PUBLIC API - High-level entry points
# ============================================================================

def load_mnist(
    data_dir: str = str(Path.home() / "datasets" / "mnist"),
    split: str = "train",
    batch_size: int = 512,
    format: str = "1d",
    normalize: bool = True,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
    if format not in ("1d", "2d"):
        raise ValueError(f"Invalid format: {format}. Must be '1d' or '2d'")

    ds = tfds.load(
        "mnist",
        split=split,
        data_dir=data_dir,
        as_supervised=True,
    )
    numpy_ds = tfds.as_numpy(ds)
    images_list, labels_list = zip(*list(numpy_ds))
    images, labels = map(np.stack, [images_list, labels_list])

    # Preprocess images
    images = _preprocess_mnist_images(images, format=format, normalize=normalize)

    n_samples = len(images)
    
    if split == "train":
        # Training: infinite stream with random sampling
        rng = np.random.default_rng(seed)
        while True:
            indices = rng.integers(0, n_samples, size=batch_size)
            yield images[indices], labels[indices]
    else:
        # Validation/test: yield all data exactly once in sequential batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            yield images[i:end_idx], labels[i:end_idx]
