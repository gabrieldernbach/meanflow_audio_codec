"""Benchmark MP3 loading libraries: np3 vs librosa vs pymp3 vs minimp3py.

Note: fast-mp3-augment is not included as it's designed for augmentation
(encoding/decoding for data augmentation), not for loading MP3 files from disk.
"""
import argparse
import time
from pathlib import Path

import numpy as np

# Try to import np3, fallback gracefully if not available
try:
    import np3
    NP3_AVAILABLE = True
except ImportError:
    NP3_AVAILABLE = False
    print("⚠️  np3 not available. Install with: pip install np3")

# Try to import pymp3, fallback gracefully if not available
try:
    import pymp3
    PYMP3_AVAILABLE = True
except ImportError:
    PYMP3_AVAILABLE = False
    print("⚠️  pymp3 not available. Install with: pip install pymp3 (requires CMake)")

# Try to import minimp3py, fallback gracefully if not available
try:
    import minimp3py
    MINIMP3PY_AVAILABLE = True
except ImportError:
    MINIMP3PY_AVAILABLE = False
    print("⚠️  minimp3py not available. Install with: pip install git+https://github.com/f0k/minimp3py.git")

import librosa


def load_audio_librosa(file: Path, sr: int = 44100) -> np.ndarray:
    """Load audio using librosa (current implementation)."""
    audio_np, _ = librosa.load(str(file), sr=sr, mono=False, dtype=np.float32)
    # librosa returns (n_samples,) for mono, (n_channels, n_samples) for stereo
    if audio_np.ndim == 1:
        # Make explicit stereo by copying mono channel to left/right
        audio = np.stack([audio_np, audio_np], axis=0)  # (2, n_samples)
    else:
        audio = audio_np
    return audio.astype(np.float32)


def load_audio_np3(file: Path, sr: int = 44100) -> np.ndarray:
    """Load audio using np3 (fast MP3 decoder)."""
    # np3.MP3 loads the file and provides samples, sample rate, and channels
    mp3 = np3.MP3(path=str(file))
    
    # np3 returns samples as int16 in shape (channels, samples)
    audio_int16 = mp3.samples  # (channels, samples)
    native_sr = mp3.hz
    n_channels = mp3.channels
    
    # Convert int16 to float32 (normalize to [-1, 1])
    audio = (audio_int16.astype(np.float32) / 32768.0)
    
    # Handle mono -> stereo conversion
    if n_channels == 1:
        audio = np.stack([audio[0], audio[0]], axis=0)  # (2, samples)
    
    # Resample if needed using librosa
    if native_sr != sr:
        # Resample each channel - let librosa determine the output size
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled = librosa.resample(
                audio[ch], orig_sr=native_sr, target_sr=sr
            ).astype(np.float32)
            resampled_channels.append(resampled)
        audio = np.stack(resampled_channels, axis=0)
    
    return audio.astype(np.float32)


def load_audio_pymp3(file: Path, sr: int = 44100) -> np.ndarray:
    """Load audio using pymp3 (MP3 decoder using libmp3lame/libmad)."""
    # pymp3.Decoder reads MP3 files frame by frame
    decoder = pymp3.Decoder(str(file))
    samples = []
    native_sr = None
    
    # Read all frames
    while True:
        frame = decoder.read()
        if frame is None:
            break
        # frame might contain (samples, channels) or just samples
        # Also get sample rate from first frame if available
        if native_sr is None:
            # Try to get sample rate from decoder if available
            try:
                native_sr = decoder.sample_rate
            except AttributeError:
                pass
        samples.append(frame)
    
    if not samples:
        raise ValueError(f"Could not decode MP3 file: {file}")
    
    # Concatenate all frames
    pcm_data = np.concatenate(samples, axis=0)
    
    # pymp3 returns PCM data - need to check the format
    # Typically returns (n_samples, n_channels) or (n_samples,) for mono
    if pcm_data.ndim == 1:
        # Mono: convert to stereo
        audio = np.stack([pcm_data, pcm_data], axis=0)  # (2, n_samples)
    else:
        # Transpose from (n_samples, n_channels) to (n_channels, n_samples)
        audio = pcm_data.T  # (n_channels, n_samples)
    
    # pymp3 returns int16 PCM, convert to float32
    audio = audio.astype(np.float32) / 32768.0
    
    # Get native sample rate if we couldn't get it from decoder
    if native_sr is None:
        try:
            from mutagen.mp3 import MP3
            audio_info = MP3(str(file))
            native_sr = audio_info.info.sample_rate
        except Exception:
            # Fallback: use librosa to get sample rate (slower, but works)
            try:
                _, native_sr = librosa.load(str(file), sr=None, mono=False, duration=0.1)
            except Exception:
                # Last resort: assume 44100
                native_sr = 44100
    
    # Resample if needed using librosa
    if native_sr != sr:
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled = librosa.resample(
                audio[ch], orig_sr=native_sr, target_sr=sr
            ).astype(np.float32)
            resampled_channels.append(resampled)
        audio = np.stack(resampled_channels, axis=0)
    
    return audio.astype(np.float32)


def load_audio_minimp3py(file: Path, sr: int = 44100) -> np.ndarray:
    """Load audio using minimp3py (fast minimp3 bindings, 2-5x faster than ffmpeg)."""
    # minimp3py.read() returns (wav, sample_rate) where wav is (n_samples, n_channels)
    wav, native_sr = minimp3py.read(str(file))
    
    # Convert from (n_samples, n_channels) to (n_channels, n_samples)
    if wav.ndim == 1:
        # Mono: convert to stereo
        audio = np.stack([wav, wav], axis=0)  # (2, n_samples)
    else:
        # Transpose from (n_samples, n_channels) to (n_channels, n_samples)
        audio = wav.T  # (n_channels, n_samples)
    
    # minimp3py already returns float32, so just ensure it's the right type
    audio = audio.astype(np.float32)
    
    # Resample if needed using librosa
    if native_sr != sr:
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled = librosa.resample(
                audio[ch], orig_sr=native_sr, target_sr=sr
            ).astype(np.float32)
            resampled_channels.append(resampled)
        audio = np.stack(resampled_channels, axis=0)
    
    return audio.astype(np.float32)


def benchmark_loader(name: str, loader_func, file: Path, sr: int, warmup: int = 3, repeats: int = 10):
    """Benchmark a loader function."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {name}")
    print("=" * 70)
    
    # Warmup
    for _ in range(warmup):
        try:
            _ = loader_func(file, sr)
        except Exception as e:
            print(f"  ❌ Error during warmup: {e}")
            return None, None, None
    
    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            audio = loader_func(file, sr)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"  ❌ Error during benchmark: {e}")
            return None, None, None
    
    if not times:
        return None, None, None
    
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    total_time = sum(times) * 1000
    
    # Get file info
    file_size_mb = file.stat().st_size / (1024 * 1024)
    
    # Load once more to get audio info
    try:
        audio = loader_func(file, sr)
        audio_duration = audio.shape[-1] / sr
        print(f"  File: {file.name}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio duration: {audio_duration:.2f} s")
        print(f"  Mean time: {mean_time:6.2f} ± {std_time:.2f} ms")
        print(f"  Total time: {total_time:6.2f} ms")
        print(f"  Throughput: {file_size_mb / (mean_time / 1000):.2f} MB/s")
        print(f"  Time per second of audio: {mean_time / audio_duration:.2f} ms/s")
    except Exception as e:
        print(f"  ⚠️  Could not get audio info: {e}")
    
    return mean_time, std_time, audio.shape if 'audio' in locals() else None


def main():
    parser = argparse.ArgumentParser(description="Benchmark MP3 loaders: np3 vs librosa")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing MP3 files",
    )
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of files to benchmark",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of benchmark repetitions",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return
    
    # Find MP3 files
    mp3_files = [f for f in data_dir.iterdir() if f.suffix.lower() == ".mp3" and f.is_file()]
    if not mp3_files:
        print(f"Error: No MP3 files found in {data_dir}")
        return
    
    mp3_files = mp3_files[: args.num_files]
    print("=" * 70)
    print("MP3 LOADER BENCHMARK: np3 vs librosa vs pymp3 vs minimp3py")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Sample rate: {args.sr}")
    print(f"Files to test: {len(mp3_files)}")
    print(f"np3 available: {NP3_AVAILABLE}")
    print(f"pymp3 available: {PYMP3_AVAILABLE}")
    print(f"minimp3py available: {MINIMP3PY_AVAILABLE}")
    
    if not NP3_AVAILABLE and not PYMP3_AVAILABLE and not MINIMP3PY_AVAILABLE:
        print("\n⚠️  No alternative MP3 loaders are installed.")
        print("   Install with: pip install np3")
        print("   Or: pip install git+https://github.com/f0k/minimp3py.git")
        print("   Or: pip install pymp3 (requires CMake)")
        print("   Benchmarking librosa only...")
    
    results_librosa = []
    results_np3 = []
    results_pymp3 = []
    results_minimp3py = []
    
    for i, file in enumerate(mp3_files, 1):
        print(f"\n{'=' * 70}")
        print(f"File {i}/{len(mp3_files)}: {file.name}")
        print("=" * 70)
        
        # Benchmark librosa
        mean_lib, std_lib, shape_lib = benchmark_loader(
            "librosa", load_audio_librosa, file, args.sr, repeats=args.repeats
        )
        if mean_lib is not None:
            results_librosa.append((file.name, mean_lib, std_lib, shape_lib))
        
        # Benchmark np3 if available
        if NP3_AVAILABLE:
            mean_np3, std_np3, shape_np3 = benchmark_loader(
                "np3", load_audio_np3, file, args.sr, repeats=args.repeats
            )
            if mean_np3 is not None:
                results_np3.append((file.name, mean_np3, std_np3, shape_np3))
                
                # Compare shapes
                if shape_lib is not None and shape_np3 is not None:
                    if shape_lib != shape_np3:
                        print(f"\n  ⚠️  Shape mismatch: librosa={shape_lib}, np3={shape_np3}")
        
        # Benchmark pymp3 if available
        if PYMP3_AVAILABLE:
            mean_pymp3, std_pymp3, shape_pymp3 = benchmark_loader(
                "pymp3", load_audio_pymp3, file, args.sr, repeats=args.repeats
            )
            if mean_pymp3 is not None:
                results_pymp3.append((file.name, mean_pymp3, std_pymp3, shape_pymp3))
                
                # Compare shapes
                if shape_lib is not None and shape_pymp3 is not None:
                    if shape_lib != shape_pymp3:
                        print(f"\n  ⚠️  Shape mismatch: librosa={shape_lib}, pymp3={shape_pymp3}")
        
        # Benchmark minimp3py if available
        if MINIMP3PY_AVAILABLE:
            mean_minimp3py, std_minimp3py, shape_minimp3py = benchmark_loader(
                "minimp3py", load_audio_minimp3py, file, args.sr, repeats=args.repeats
            )
            if mean_minimp3py is not None:
                results_minimp3py.append((file.name, mean_minimp3py, std_minimp3py, shape_minimp3py))
                
                # Compare shapes
                if shape_lib is not None and shape_minimp3py is not None:
                    if shape_lib != shape_minimp3py:
                        print(f"\n  ⚠️  Shape mismatch: librosa={shape_lib}, minimp3py={shape_minimp3py}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results_librosa:
        lib_times = [r[1] for r in results_librosa]
        lib_mean = np.mean(lib_times)
        lib_std = np.std(lib_times)
        print(f"\nlibrosa:")
        print(f"  Mean: {lib_mean:6.2f} ± {lib_std:.2f} ms")
        print(f"  Per file: {lib_mean / len(results_librosa):6.2f} ms")
    
    if results_np3:
        np3_times = [r[1] for r in results_np3]
        np3_mean = np.mean(np3_times)
        np3_std = np.std(np3_times)
        print(f"\nnp3:")
        print(f"  Mean: {np3_mean:6.2f} ± {np3_std:.2f} ms")
        print(f"  Per file: {np3_mean / len(results_np3):6.2f} ms")
    
    if results_pymp3:
        pymp3_times = [r[1] for r in results_pymp3]
        pymp3_mean = np.mean(pymp3_times)
        pymp3_std = np.std(pymp3_times)
        print(f"\npymp3:")
        print(f"  Mean: {pymp3_mean:6.2f} ± {pymp3_std:.2f} ms")
        print(f"  Per file: {pymp3_mean / len(results_pymp3):6.2f} ms")
    
    if results_minimp3py:
        minimp3py_times = [r[1] for r in results_minimp3py]
        minimp3py_mean = np.mean(minimp3py_times)
        minimp3py_std = np.std(minimp3py_times)
        print(f"\nminimp3py:")
        print(f"  Mean: {minimp3py_mean:6.2f} ± {minimp3py_std:.2f} ms")
        print(f"  Per file: {minimp3py_mean / len(results_minimp3py):6.2f} ms")
    
    # Compare all available libraries
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    
    if results_librosa and results_np3:
        speedup = lib_mean / np3_mean
        print(f"\nnp3 vs librosa: np3 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than librosa")
    
    if results_librosa and results_pymp3:
        speedup = lib_mean / pymp3_mean
        print(f"pymp3 vs librosa: pymp3 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than librosa")
    
    if results_np3 and results_pymp3:
        speedup = np3_mean / pymp3_mean
        print(f"pymp3 vs np3: pymp3 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than np3")
    
    if results_librosa and results_minimp3py:
        speedup = lib_mean / minimp3py_mean
        print(f"minimp3py vs librosa: minimp3py is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than librosa")
    
    if results_np3 and results_minimp3py:
        speedup = np3_mean / minimp3py_mean
        print(f"minimp3py vs np3: minimp3py is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than np3")
    
    # Per-file comparison table
    if results_librosa:
        print(f"\n{'File':<30} {'librosa (ms)':<15}", end="")
        if results_np3:
            print(f" {'np3 (ms)':<15}", end="")
        if results_pymp3:
            print(f" {'pymp3 (ms)':<15}", end="")
        if results_minimp3py:
            print(f" {'minimp3py (ms)':<15}", end="")
        print()
        print("-" * 105)
        
        # Create a dict to match files across results
        all_files = set()
        if results_librosa:
            all_files.update(r[0] for r in results_librosa)
        if results_np3:
            all_files.update(r[0] for r in results_np3)
        if results_pymp3:
            all_files.update(r[0] for r in results_pymp3)
        if results_minimp3py:
            all_files.update(r[0] for r in results_minimp3py)
        
        lib_dict = {r[0]: r[1] for r in results_librosa}
        np3_dict = {r[0]: r[1] for r in results_np3} if results_np3 else {}
        pymp3_dict = {r[0]: r[1] for r in results_pymp3} if results_pymp3 else {}
        minimp3py_dict = {r[0]: r[1] for r in results_minimp3py} if results_minimp3py else {}
        
        for filename in sorted(all_files):
            line = f"{filename:<30}"
            if filename in lib_dict:
                line += f" {lib_dict[filename]:>13.2f}"
            else:
                line += " " * 15
            if filename in np3_dict:
                line += f" {np3_dict[filename]:>13.2f}"
            elif results_np3:
                line += " " * 15
            if filename in pymp3_dict:
                line += f" {pymp3_dict[filename]:>13.2f}"
            elif results_pymp3:
                line += " " * 15
            if filename in minimp3py_dict:
                line += f" {minimp3py_dict[filename]:>13.2f}"
            elif results_minimp3py:
                line += " " * 15
            print(line)


if __name__ == "__main__":
    main()

