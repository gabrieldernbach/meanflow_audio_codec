"""
Benchmark prefetch vs no-prefetch for audio loading.
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from meanflow_audio_codec.datasets.audio import glob_audio_files, load_audio


def create_test_files(tmp_path: Path, n_files: int, duration: float, sr: int):
    """Create test audio files."""
    for i in range(n_files):
        audio = np.random.randn(int(sr * duration)).astype(np.float32)
        sf.write(str(tmp_path / f"audio_{i:04d}.wav"), audio, sr)


def benchmark_load(prefetch: bool, n_files: int, frame_sz: int):
    """Benchmark audio loading with or without prefetch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_test_files(tmp_path, n_files, duration=1.0, sr=44100)

        files = glob_audio_files(str(tmp_path), seed=42)

        start = time.time()
        frames = list(load_audio(files, seed=42, frame_sz=frame_sz, prefetch=prefetch))
        elapsed = time.time() - start

        n_frames = len(frames)
        mode = "prefetch" if prefetch else "no-prefetch"
        print(f"  {mode:12} ({n_files} files, {n_frames} frames): "
              f"{elapsed*1000:.2f} ms ({n_frames/elapsed:.1f} frames/s)")
        return elapsed


def main():
    print("=" * 70)
    print("Prefetch vs No-Prefetch Benchmark")
    print("=" * 70)

    print("\nComparing prefetch enabled vs disabled:")
    for n_files in [10, 50, 100]:
        print(f"\n{n_files} files:")
        benchmark_load(prefetch=False, n_files=n_files, frame_sz=1024)
        benchmark_load(prefetch=True, n_files=n_files, frame_sz=1024)

    print("\n" + "=" * 70)
    print("Note: Prefetch helps mask I/O delays when switching between files")
    print("=" * 70)


if __name__ == "__main__":
    main()

