import logging
import random
import threading
from collections import deque
from pathlib import Path
from typing import Iterator, TypeVar

import numpy as np
from toolz import compose, curry

logger = logging.getLogger(__name__)

# Try to import minimp3py for fast MP3 loading (optional)
try:
    import minimp3py
    MINIMP3PY_AVAILABLE = True
except ImportError:
    MINIMP3PY_AVAILABLE = False
    logger.warning(
        "minimp3py not available. MP3 loading will be slower. "
        "Install with: uv sync --extra audio or "
        "CFLAGS='-O3 -march=native' pip install "
        "git+https://github.com/f0k/minimp3py.git"
    )

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# PUBLIC API - High-level entry points
# ============================================================================

def build_audio_pipeline(
    data_dir: str,
    seed: int,
    frame_sz: int = 256 * 256 * 3,
    prefetch: int = 4,
    buffer_size: int = 1000,
    batch_size: int = 32,
    drop_last: bool = False,
) -> Iterator[np.ndarray]:
    """Build complete audio data pipeline from configuration.

    Args:
        data_dir: Directory containing audio files
        seed: Random seed for shuffling and padding
        frame_sz: Size of each frame in samples
        prefetch: Number of files to prefetch ahead
        buffer_size: Size of shuffle buffer
        batch_size: Batch size
        drop_last: Whether to drop last incomplete batch

    Yields:
        Batched audio frames with shape (batch_size, frame_sz, n_channels)
    """
    pipeline = compose(
        batch(batch_size=batch_size, drop_last=drop_last),
        buffer_shuffle(buffer_size=buffer_size, seed=seed),
        audio_to_frames(frame_sz=frame_sz, seed=seed),
        load_audio_files(prefetch=prefetch),
    )
    files = glob_audio_files(data_dir, seed=seed)
    return pipeline(iter(files))


def load_audio(
    files: list[Path],
    seed: int,
    frame_sz: int = 256 * 256 * 3,
    prefetch: int = 4,
) -> Iterator[np.ndarray]:
    """Convenience function: load audio files and yield frames."""
    audio_gen = load_audio_files(iter(files), prefetch=prefetch)
    return audio_to_frames(audio_gen, frame_sz=frame_sz, seed=seed)


# ============================================================================
# COMPOSABLE FUNCTIONS - Intermediate abstractions
# ============================================================================

def glob_audio_files(data_dir: str, seed: int) -> list[Path]:
    """Discover and shuffle MP3 files in directory."""
    files = [
        f for f in Path(data_dir).iterdir()
        if f.suffix.lower() == ".mp3" and f.is_file()
    ]
    random.Random(seed).shuffle(files)
    return files


@curry
def load_audio_files(
    files: Iterator[Path],
    prefetch: int = 4,
) -> Iterator[np.ndarray]:
    """Load MP3 audio files with optional prefetching.

    Only yields audio files that are 44.1kHz. Non-44.1kHz files are dropped
    with a warning.
    """
    if prefetch <= 0:
        for file in files:
            try:
                audio = _load_audio(file)
                if audio is not None:
                    yield audio
            except Exception:
                continue
        return

    queue = deque(maxlen=prefetch * 2)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_prefetch_worker,
        args=(files, queue, stop_event),
        daemon=True,
    )
    thread.start()

    try:
        while thread.is_alive() or queue:
            if queue:
                audio = queue.popleft()
                if audio is not None:
                    yield audio
            else:
                threading.Event().wait(0.001)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


@curry
def audio_to_frames(
    audio_files: Iterator[np.ndarray],
    frame_sz: int,
    seed: int,
) -> Iterator[np.ndarray]:
    """Convert audio files into fixed-size frames.

    Yields frames with shape (frame_sz, n_channels) for compatibility
    with MDCT layers that expect [B, T, C] format.
    """
    rng = np.random.default_rng(seed)

    for audio in audio_files:
        padded_audio = _prepend_and_pad_audio(audio, frame_sz, rng)
        n_frames = padded_audio.shape[-1] // frame_sz
        n_channels = padded_audio.shape[0]

        # Transpose first: (n_channels, n_samples) -> (n_samples, n_channels)
        # Then reshape: (n_samples, n_channels) -> (n_frames, frame_sz,
        # n_channels)
        frames = padded_audio.transpose(1, 0).reshape(
            n_frames,
            frame_sz,
            n_channels,
        )

        for frame in frames:
            yield frame


@curry
def buffer_shuffle(
    generator: Iterator[T],
    buffer_size: int,
    seed: int,
) -> Iterator[T]:
    """Shuffle items from generator using a buffer."""
    rng = np.random.default_rng(seed)
    buffer = []

    for item in generator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            # Swap-and-pop: O(1) removal instead of O(n) for list.pop()
            yield _swap_and_pop(buffer, rng)

    while buffer:
        yield _swap_and_pop(buffer, rng)


@curry
def batch(
    generator: Iterator[T],
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[np.ndarray]:
    """Batch items from generator."""
    batch_list = []

    for item in generator:
        batch_list.append(item)
        if len(batch_list) >= batch_size:
            yield np.stack(batch_list[:batch_size])
            batch_list = batch_list[batch_size:]

    if batch_list and not drop_last:
        yield np.stack(batch_list)


# ============================================================================
# PRIVATE IMPLEMENTATION - Low-level helpers
# ============================================================================

def _swap_and_pop(buffer: list, rng: np.random.Generator) -> T:
    """Remove and return a random element from buffer using swap-and-pop.

    This is O(1) instead of O(n) for list.pop(idx) at a random index.
    """
    if len(buffer) == 1:
        return buffer.pop()
    idx = rng.integers(0, len(buffer) - 1)
    buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
    return buffer.pop()


def _load_audio(file: Path) -> np.ndarray | None:
    """Load MP3 file and return float32 array preserving channels.

    Only supports MP3 files. Drops audio that is not 44.1kHz.
    Returns audio with shape (n_channels, n_samples) or None if dropped.

    Args:
        file: Path to MP3 file

    Returns:
        Audio array with shape (n_channels, n_samples) or None if dropped
    """
    if not MINIMP3PY_AVAILABLE:
        raise ImportError(
            "minimp3py is required for audio loading. "
            "Install with: uv sync --extra audio or "
            "CFLAGS='-O3 -march=native' pip install "
            "git+https://github.com/f0k/minimp3py.git"
        )

    # minimp3py.read() returns (wav, sample_rate)
    # wav is (n_samples, n_channels)
    wav, native_sr = minimp3py.read(str(file))

    # Drop audio that is not 44.1kHz
    if native_sr != 44100:
        logger.warning(
            f"Dropping audio file {file.name}: "
            f"sample rate is {native_sr}Hz, expected 44100Hz"
        )
        return None

    # Convert from (n_samples, n_channels) to (n_channels, n_samples)
    if wav.ndim == 1:
        # Mono: convert to stereo
        audio = np.stack([wav, wav], axis=0)  # (2, n_samples)
    else:
        # Transpose from (n_samples, n_channels) to (n_channels, n_samples)
        audio = wav.T  # (n_channels, n_samples)

    # minimp3py already returns float32
    return audio.astype(np.float32)


def _prepend_and_pad_audio(
    audio: np.ndarray, frame_sz: int, rng: np.random.Generator
) -> np.ndarray:
    """Prepend random padding and pad to frame size boundary."""
    n_prepend = rng.integers(0, frame_sz + 1)
    current_length = audio.shape[-1]
    length_after_prepend = current_length + n_prepend
    remainder = length_after_prepend % frame_sz
    n_postpend = (frame_sz - remainder) % frame_sz

    if n_prepend > 0 or n_postpend > 0:
        pad_width = ((0, 0), (n_prepend, n_postpend))
        audio = np.pad(audio, pad_width, mode='constant', constant_values=0)
    return audio


def _prefetch_worker(
    file_gen: Iterator[Path],
    queue: deque,
    stop_event: threading.Event,
):
    """Background worker thread for prefetching MP3 audio files."""
    for file in file_gen:
        if stop_event.is_set():
            break
        try:
            audio = _load_audio(file)
            # Only append if audio is not None (not dropped)
            if audio is not None:
                queue.append(audio)
        except Exception:
            continue
