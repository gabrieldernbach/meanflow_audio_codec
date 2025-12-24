"""Benchmark components of audio.py to identify performance bottlenecks."""
import argparse
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from meanflow_audio_codec.datasets.audio import (_load_audio, _prepend_and_pad_audio,
                                          audio_to_frames, batch, buffer_shuffle,
                                          build_audio_pipeline,
                                          glob_audio_files,
                                          load_audio_files)


def benchmark_component(name: str, func, *args, warmup: int = 3, repeats: int = 10):
    """Benchmark a function with warmup and averaging."""
    # Warmup
    for _ in range(warmup):
        try:
            result = func(*args)
            # Consume generator if needed
            if isinstance(result, Iterator):
                list(result)
        except Exception as e:
            print(f"  ⚠️  Warning: {name} failed during warmup: {e}")
            return None, None, None
    
    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            result = func(*args)
            # Consume generator if needed
            if isinstance(result, Iterator):
                list(result)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"  ⚠️  Warning: {name} failed during benchmark: {e}")
            return None, None, None
    
    if not times:
        return None, None, None
    
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    total_time = sum(times) * 1000
    
    return mean_time, std_time, total_time


def benchmark_glob_files(data_dir: str, seed: int, warmup: int = 3, repeats: int = 10):
    """Benchmark file globbing."""
    print("\n" + "=" * 70)
    print("1. BENCHMARK: glob_audio_files")
    print("=" * 70)
    
    mean, std, total = benchmark_component(
        "glob_audio_files",
        glob_audio_files,
        data_dir,
        seed,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        files = glob_audio_files(data_dir, seed)
        print(f"  Files found: {len(files)}")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        return mean, len(files)
    return None, 0


def benchmark_load_audio_single(file: Path, warmup: int = 3, repeats: int = 10):
    """Benchmark loading a single audio file."""
    print("\n" + "=" * 70)
    print("2. BENCHMARK: _load_audio (single file)")
    print("=" * 70)
    
    if not file.exists():
        print(f"  ⚠️  File not found: {file}")
        return None
    
    mean, std, total = benchmark_component(
        "_load_audio",
        _load_audio,
        file,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        audio = _load_audio(file)
        file_size_mb = file.stat().st_size / (1024 * 1024)
        audio_duration = audio.shape[-1] / 44100
        print(f"  File: {file.name}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio duration: {audio_duration:.2f} s")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        print(f"  Throughput: {file_size_mb / (mean / 1000):.2f} MB/s")
        return mean, audio.shape
    return None, None


def benchmark_load_audio_files(
    files: list[Path], prefetch: int, warmup: int = 1, repeats: int = 3
):
    """Benchmark loading multiple audio files with prefetching."""
    print("\n" + "=" * 70)
    print(f"3. BENCHMARK: load_audio_files (prefetch={prefetch})")
    print("=" * 70)
    
    def load_all():
        gen = load_audio_files(iter(files), prefetch=prefetch)
        return list(gen)
    
    mean, std, total = benchmark_component(
        "load_audio_files",
        load_all,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  Files: {len(files)}")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        print(f"  Throughput: {total_size / (mean / 1000):.2f} MB/s")
        print(f"  Time per file: {mean / len(files):6.2f} ms")
        return mean, len(files)
    return None, 0


def benchmark_prepend_and_pad(
    audio: np.ndarray, frame_sz: int, seed: int, warmup: int = 3, repeats: int = 100
):
    """Benchmark prepend and pad operation."""
    print("\n" + "=" * 70)
    print("4. BENCHMARK: prepend_and_pad_audio")
    print("=" * 70)
    
    rng = np.random.default_rng(seed)
    
    def pad_audio():
        return _prepend_and_pad_audio(audio, frame_sz, rng)
    
    mean, std, total = benchmark_component(
        "prepend_and_pad_audio",
        pad_audio,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        print(f"  Input shape: {audio.shape}")
        print(f"  Frame size: {frame_sz}")
        result = _prepend_and_pad_audio(audio, frame_sz, rng)
        print(f"  Output shape: {result.shape}")
        print(f"  Mean time: {mean:6.4f} ± {std:.4f} ms")
        print(f"  Total time: {total:6.2f} ms")
        return mean
    return None


def benchmark_audio_to_frames(
    audio: np.ndarray, frame_sz: int, seed: int, warmup: int = 3, repeats: int = 10
):
    """Benchmark audio to frames conversion."""
    print("\n" + "=" * 70)
    print("5. BENCHMARK: audio_to_frames")
    print("=" * 70)
    
    def frames_gen():
        gen = audio_to_frames(iter([audio]), frame_sz=frame_sz, seed=seed)
        return list(gen)
    
    mean, std, total = benchmark_component(
        "audio_to_frames",
        frames_gen,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        frames = list(audio_to_frames(iter([audio]), frame_sz=frame_sz, seed=seed))
        print(f"  Input shape: {audio.shape}")
        print(f"  Frame size: {frame_sz}")
        print(f"  Frames generated: {len(frames)}")
        print(f"  Frame shape: {frames[0].shape if frames else 'N/A'}")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        if len(frames) > 0:
            print(f"  Time per frame: {mean / len(frames):6.4f} ms")
        return mean, len(frames)
    return None, 0


def benchmark_buffer_shuffle(
    items: list, buffer_size: int, seed: int, warmup: int = 3, repeats: int = 10
):
    """Benchmark buffer shuffling."""
    print("\n" + "=" * 70)
    print(f"6. BENCHMARK: buffer_shuffle (buffer_size={buffer_size})")
    print("=" * 70)
    
    def shuffle_all():
        gen = buffer_shuffle(iter(items), buffer_size=buffer_size, seed=seed)
        return list(gen)
    
    mean, std, total = benchmark_component(
        "buffer_shuffle",
        shuffle_all,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        print(f"  Items: {len(items)}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        print(f"  Time per item: {mean / len(items):6.4f} ms")
        return mean
    return None


def benchmark_batch(
    items: list, batch_size: int, warmup: int = 3, repeats: int = 10
):
    """Benchmark batching."""
    print("\n" + "=" * 70)
    print(f"7. BENCHMARK: batch (batch_size={batch_size})")
    print("=" * 70)
    
    def batch_all():
        gen = batch(iter(items), batch_size=batch_size, drop_last=False)
        return list(gen)
    
    mean, std, total = benchmark_component(
        "batch",
        batch_all,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        batches = list(batch(iter(items), batch_size=batch_size, drop_last=False))
        print(f"  Items: {len(items)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches: {len(batches)}")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        print(f"  Time per batch: {mean / len(batches) if batches else 0:6.4f} ms")
        return mean
    return None


def benchmark_full_pipeline(
    data_dir: str,
    seed: int,
    frame_sz: int,
    prefetch: int,
    buffer_size: int,
    batch_size: int,
    num_batches: int = 5,
    warmup: int = 1,
    repeats: int = 3,
):
    """Benchmark the full pipeline end-to-end."""
    print("\n" + "=" * 70)
    print("8. BENCHMARK: Full Pipeline (end-to-end)")
    print("=" * 70)
    
    def run_pipeline():
        pipeline = build_audio_pipeline(
            data_dir=data_dir,
            seed=seed,
            frame_sz=frame_sz,
            prefetch=prefetch,
            buffer_size=buffer_size,
            batch_size=batch_size,
            drop_last=False,
        )
        batches = []
        for i, batch_data in enumerate(pipeline):
            batches.append(batch_data)
            if i >= num_batches - 1:
                break
        return batches
    
    mean, std, total = benchmark_component(
        "full_pipeline",
        run_pipeline,
        warmup=warmup,
        repeats=repeats,
    )
    
    if mean is not None:
        batches = run_pipeline()
        total_samples = sum(b.shape[0] for b in batches)
        print(f"  Batches processed: {len(batches)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Mean time: {mean:6.2f} ± {std:.2f} ms")
        print(f"  Total time: {total:6.2f} ms")
        print(f"  Time per batch: {mean / len(batches):6.2f} ms")
        print(f"  Time per sample: {mean / total_samples:6.4f} ms")
        return mean
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark audio.py components")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--frame-sz", type=int, default=256 * 256 * 3, help="Frame size"
    )
    parser.add_argument("--prefetch", type=int, default=4, help="Prefetch count")
    parser.add_argument(
        "--buffer-size", type=int, default=1000, help="Shuffle buffer size"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full pipeline benchmark",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return
    
    print("=" * 70)
    print("AUDIO_SIMPLE.PY COMPONENT BENCHMARK")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print("Sample rate: 44100 (fixed)")
    print(f"Frame size: {args.frame_sz}")
    print(f"Prefetch: {args.prefetch}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    
    results = {}
    
    # 1. Benchmark file globbing
    glob_time, num_files = benchmark_glob_files(args.data_dir, args.seed)
    results["glob"] = glob_time
    if num_files == 0:
        print("\n⚠️  No audio files found. Exiting.")
        return
    
    # Get files for subsequent benchmarks
    files = glob_audio_files(args.data_dir, args.seed)
    if args.num_files:
        files = files[: args.num_files]
        print(f"\n⚠️  Limiting to {len(files)} files for benchmarking")
    
    # 2. Benchmark loading a single file
    if files:
        load_single_time, audio_shape = benchmark_load_audio_single(
            files[0]
        )
        results["load_single"] = load_single_time
        
        # Load a sample audio for frame processing benchmarks
        sample_audio = _load_audio(files[0])
    else:
        sample_audio = None
    
    # 3. Benchmark loading multiple files
    if len(files) > 1:
        load_files_time, _ = benchmark_load_audio_files(
            files[: min(10, len(files))], args.prefetch
        )
        results["load_files"] = load_files_time
    
    # 4. Benchmark prepend and pad
    if sample_audio is not None:
        pad_time = benchmark_prepend_and_pad(sample_audio, args.frame_sz, args.seed)
        results["prepend_pad"] = pad_time
    
    # 5. Benchmark audio to frames
    if sample_audio is not None:
        frames_time, num_frames = benchmark_audio_to_frames(
            sample_audio, args.frame_sz, args.seed
        )
        results["audio_to_frames"] = frames_time
    
    # 6. Benchmark buffer shuffle (with dummy frames)
    if sample_audio is not None:
        frames = list(
            audio_to_frames(iter([sample_audio]), frame_sz=args.frame_sz, seed=args.seed)
        )
        shuffle_time = benchmark_buffer_shuffle(
            frames[: min(100, len(frames))], args.buffer_size, args.seed
        )
        results["buffer_shuffle"] = shuffle_time
    
    # 7. Benchmark batching
    if sample_audio is not None:
        frames = list(
            audio_to_frames(iter([sample_audio]), frame_sz=args.frame_sz, seed=args.seed)
        )
        batch_time = benchmark_batch(frames[: 50], args.batch_size)
        results["batch"] = batch_time
    
    # 8. Benchmark full pipeline
    if not args.skip_full:
        full_pipeline_time = benchmark_full_pipeline(
            args.data_dir,
            args.seed,
            args.frame_sz,
            args.prefetch,
            args.buffer_size,
            args.batch_size,
            num_batches=5,
        )
        results["full_pipeline"] = full_pipeline_time
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Component Timings")
    print("=" * 70)
    print(f"{'Component':<25} {'Time (ms)':<15} {'% of Total':<15}")
    print("-" * 70)
    
    total_time = sum(v for v in results.values() if v is not None)
    
    for component, time_ms in sorted(results.items(), key=lambda x: x[1] or 0, reverse=True):
        if time_ms is not None:
            percentage = (time_ms / total_time * 100) if total_time > 0 else 0
            print(f"{component:<25} {time_ms:>12.2f} ms {percentage:>13.2f}%")
        else:
            print(f"{component:<25} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 70)
    print(f"{'Total (sum of components)':<25} {total_time:>12.2f} ms")
    
    # Identify bottleneck (excluding full_pipeline from individual component analysis)
    if results:
        valid_results = {k: v for k, v in results.items() if v is not None and k != "full_pipeline"}
        if valid_results:
            bottleneck = max(valid_results.items(), key=lambda x: x[1])
            print("\n" + "=" * 70)
            print("🔍 PERFORMANCE BOTTLENECK (Individual Components)")
            print("=" * 70)
            print(f"Component: {bottleneck[0]}")
            print(f"Time: {bottleneck[1]:.2f} ms")
            component_total = sum(v for v in valid_results.values())
            if component_total > 0:
                print(f"Percentage of component time: {bottleneck[1] / component_total * 100:.2f}%")
            
            # Additional insights
            if bottleneck[0] == "load_single" or bottleneck[0] == "load_files":
                print("\n💡 Insight: Audio loading is the primary bottleneck.")
                print("   Analysis:")
                if "load_single" in results and results["load_single"]:
                    print(f"   - Single file load: {results['load_single']:.2f} ms")
                if "load_files" in results and results["load_files"]:
                    num_files = len(files) if 'files' in locals() else 1
                    print(f"   - Multi-file load: {results['load_files']:.2f} ms ({results['load_files']/num_files:.2f} ms/file)")
                print("   Recommendations:")
                print("   1. Use soundfile instead of librosa for faster loading")
                print("   2. Pre-process audio to uncompressed format (WAV)")
                print("   3. Increase prefetch count to overlap I/O with processing")
                print("   4. Consider parallel loading with multiprocessing")
            elif bottleneck[0] == "audio_to_frames":
                print("\n💡 Insight: Frame conversion is the bottleneck.")
                print("   Consider optimizing reshape/array operations.")
            elif bottleneck[0] == "buffer_shuffle":
                print("\n💡 Insight: Shuffling is the bottleneck.")
                print("   Consider reducing buffer_size or optimizing shuffle logic.")
        
        # Full pipeline analysis
        if "full_pipeline" in results and results["full_pipeline"]:
            print("\n" + "=" * 70)
            print("📊 FULL PIPELINE ANALYSIS")
            print("=" * 70)
            full_time = results["full_pipeline"]
            component_sum = sum(v for k, v in results.items() if v is not None and k != "full_pipeline")
            overhead = full_time - component_sum
            print(f"Full pipeline time: {full_time:.2f} ms")
            print(f"Sum of components: {component_sum:.2f} ms")
            print(f"Estimated overhead: {overhead:.2f} ms ({overhead/full_time*100:.1f}%)")
            print("\n   Note: Full pipeline includes:")
            print("   - Multiple files (not just one)")
            print("   - Complete processing chain")
            print("   - Generator overhead")
            print("   - Thread synchronization (prefetch)")


if __name__ == "__main__":
    main()

