"""Benchmark audio dataloader implementations and identify bottlenecks."""
import os
import signal
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from meanflow_audio_codec.datasets.audio import (load_audio_train,
                                          load_audio_train_grain)


class TimeoutError(Exception):
    """Raised when a benchmark operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Benchmark operation timed out")


def benchmark_loader(
    loader_fn: Callable[..., Iterator[np.ndarray]],
    num_batches: int = 50,
    timeout_seconds: int = 300,  # 5 minute default timeout
    **loader_kwargs,
) -> dict:
    """Benchmark a data loader function with timeout protection."""
    print(f"\nBenchmarking {loader_fn.__name__}...")
    print(f"  Config: {loader_kwargs}")
    print(f"  Timeout: {timeout_seconds}s, Max batches: {num_batches}")
    
    overall_start = time.time()
    
    try:
        # Time the first batch (cold start) with timeout check
        iterator_start = time.time()
        iterator = loader_fn(**loader_kwargs)
        start = time.time()
        
        # Check timeout before first batch
        if time.time() - overall_start > timeout_seconds:
            print(f"  ❌ TIMEOUT: Exceeded {timeout_seconds}s before first batch")
            return {"error": "timeout", "timeout_seconds": timeout_seconds}
        
        first_batch = next(iterator)
        first_batch_time = time.time() - start
        
        print(f"  First batch time: {first_batch_time:.3f}s")
        print(f"  First batch shape: {first_batch.shape}, dtype: {first_batch.dtype}")
        
        # Time subsequent batches with iteration limit and timeout
        batch_times = []
        total_samples = 0
        
        start = time.time()
        for i, batch in enumerate(iterator):
            # Check timeout on each iteration
            elapsed = time.time() - overall_start
            if elapsed > timeout_seconds:
                print(f"  ⚠️  Timeout reached after {elapsed:.1f}s, stopping")
                break
            
            batch_times.append(time.time() - start)
            total_samples += batch.shape[0]
            start = time.time()
            
            # Hard limit on number of batches
            if i + 1 >= num_batches:
                break
                
    except StopIteration:
        print(f"  ⚠️  Iterator exhausted before {num_batches} batches")
    except Exception as e:
        print(f"  ❌ ERROR during benchmarking: {e}")
        return {"error": str(e)}
    
    if not batch_times:
        elapsed = time.time() - overall_start
        print(f"  ⚠️  No batches produced (elapsed: {elapsed:.1f}s)")
        return {"error": "No batches produced", "elapsed_seconds": elapsed}
    
    # Check if we hit timeout or max batches
    if len(batch_times) < num_batches:
        print(f"  Note: Only collected {len(batch_times)}/{num_batches} batches")
    
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    min_batch_time = np.min(batch_times)
    max_batch_time = np.max(batch_times)
    
    # Calculate throughput
    samples_per_sec = loader_kwargs.get("batch_size", 32) / avg_batch_time
    total_time = sum(batch_times)
    
    results = {
        "first_batch_time": first_batch_time,
        "avg_batch_time": avg_batch_time,
        "std_batch_time": std_batch_time,
        "min_batch_time": min_batch_time,
        "max_batch_time": max_batch_time,
        "total_samples": total_samples,
        "total_time": total_time,
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": 1.0 / avg_batch_time,
    }
    
    print(f"  Average batch time: {avg_batch_time:.3f}s ± {std_batch_time:.3f}s")
    print(f"  Batch time range: [{min_batch_time:.3f}s, {max_batch_time:.3f}s]")
    print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"  Total time: {total_time:.2f}s for {len(batch_times)} batches")
    
    return results


def profile_memory_and_io(timeout_seconds: int = 60):
    """Profile memory usage and I/O patterns with timeout."""
    try:
        import os

        import psutil
    except ImportError:
        print("  Skipping memory profiling (psutil not available)")
        return
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\nMemory profiling (timeout: {timeout_seconds}s):")
    print(f"  Baseline memory: {baseline_memory:.2f} MB")
    
    start_time = time.time()
    try:
        # Profile grain loader
        iterator = load_audio_train_grain(batch_size=16, samples_per_frame=44100)
        
        # Get a few batches and measure memory
        for i, batch in enumerate(iterator):
            if time.time() - start_time > timeout_seconds:
                print(f"  ⚠️  Timeout reached during memory profiling")
                break
                
            current_memory = process.memory_info().rss / 1024 / 1024
            delta = current_memory - baseline_memory
            print(f"  After batch {i+1}: {current_memory:.2f} MB (+{delta:.2f} MB)")
            if i >= 4:
                break
    except Exception as e:
        print(f"  ❌ Error during memory profiling: {e}")


def identify_bottlenecks():
    """Identify specific bottlenecks in the implementation."""
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    # Resolve data directory - no silent fallbacks
    data_dir = os.environ.get("AUDIO_DATA_DIR")
    if data_dir is None:
        print(f"\nERROR: AUDIO_DATA_DIR environment variable not set")
        print("Please set AUDIO_DATA_DIR environment variable to point to audio data directory")
        return
    
    downloads_dir = Path(data_dir)
    if not downloads_dir.exists():
        print(f"\nERROR: Audio data directory not found: {downloads_dir}")
        print("Please ensure the directory specified by AUDIO_DATA_DIR exists")
        return
    
    # Count audio files
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = [f for f in downloads_dir.iterdir() 
             if f.suffix.lower() in audio_extensions and f.is_file()]
    
    print(f"\nFound {len(files)} audio files")
    if len(files) > 0:
        sample_file = files[0]
        file_size_mb = sample_file.stat().st_size / 1024 / 1024
        print(f"Sample file: {sample_file.name} ({file_size_mb:.2f} MB)")
        
        # Estimate file loading time
        import time
        start = time.time()
        try:
            import librosa
            librosa.load(str(sample_file), sr=44100, mono=False)
            load_time = time.time() - start
            print(f"  File load time: {load_time:.3f}s")
        except:
            pass
    
    print("\n1. SINGLE-THREADED LOADER (load_audio_train):")
    print("   - No multi-processing support (single-threaded)")
    print("   - Sequential file loading (I/O bound)")
    print("   - Sequential frame extraction")
    print("   - BOTTLENECK: librosa.load() is CPU-bound, blocks on each file")
    print("   - Cold start: First batch loads entire first audio file")
    print("   - Streaming shuffle buffer helps randomization but doesn't parallelize")
    
    print("\n2. GRAIN LOADER (load_audio_train_grain):")
    print("   - FIXED: Now uses file caching (waveforms cached in file_info)")
    print("   - Pre-computes all frame indices upfront (memory intensive)")
    print("   - All files loaded once during initialization")
    print("   - BOTTLENECK: Sequential initialization - loads ALL files upfront")
    print("   - BOTTLENECK: No multi-processing - Grain not configured for parallel I/O")
    print("   - Memory: Caches all waveforms in memory (can be large)")
    
    print("\n3. MULTI-PROCESSING STATUS:")
    print("   ❌ Neither implementation supports multi-processing")
    print("   ❌ No parallel file I/O")
    print("   ❌ No worker processes")
    print("   ⚠️  Grain supports parallelization but requires configuration")
    
    print("\n4. PERFORMANCE BOTTLENECKS:")
    print("   a) I/O: librosa.load() is CPU-bound and sequential")
    print("   b) Memory: Grain loader keeps all waveforms in RAM")
    print("   c) Initialization: Grain loader loads all files before first batch")
    print("   d) No caching across epochs (files reloaded each time)")
    print("   e) No prefetching/background loading")
    
    print("\n5. RECOMMENDED IMPROVEMENTS:")
    print("   a) ✅ FIXED: File caching in grain loader (prevents double-loading)")
    print("   b) 🔧 Add multiprocessing for parallel file I/O (use 'spawn' method)")
    print("   c) 🔧 Pre-process/extract frames to disk (avoid runtime I/O)")
    print("   d) 🔧 Configure Grain for parallel data loading")
    print("   e) 🔧 Add prefetching/background loading queue")
    print("   f) 🔧 Consider memory-mapped arrays for large datasets")
    print("   g) 🔧 Use jax-dataloader or loadax for JAX-native parallel loading")


def main():
    """Run comprehensive benchmarks with timeout protection."""
    print("="*80)
    print("AUDIO DATALOADER BENCHMARK")
    print("="*80)
    
    # Configuration
    BENCHMARK_TIMEOUT = 300  # 5 minutes per loader
    MEMORY_PROFILE_TIMEOUT = 60  # 1 minute for memory profiling
    NUM_BENCHMARK_BATCHES = 20  # Limit number of batches
    
    # Resolve data directory - no silent fallbacks
    data_dir = os.environ.get("AUDIO_DATA_DIR")
    if data_dir is None:
        print(f"\nERROR: AUDIO_DATA_DIR environment variable not set")
        print("Skipping benchmarks. Please set AUDIO_DATA_DIR environment variable.")
        identify_bottlenecks()
        return
    
    downloads_dir = Path(data_dir)
    if not downloads_dir.exists():
        print(f"\nERROR: Audio data directory not found: {downloads_dir}")
        print("Skipping benchmarks. Please set AUDIO_DATA_DIR or ensure ~/datasets/wavegen/ exists.")
        identify_bottlenecks()
        return
    
    # Small frame size for faster benchmarking
    frame_size = 44100  # 1 second at 44.1kHz
    batch_size = 16
    
    config = {
        "batch_size": batch_size,
        "samples_per_frame": frame_size,
        "hop_size": frame_size // 2,
        "shuffle_buffer_size": 256,
        "seed": 42,
    }
    
    print("\n" + "="*80)
    print("THROUGHPUT BENCHMARKS")
    print("="*80)
    print(f"Timeout per loader: {BENCHMARK_TIMEOUT}s")
    print(f"Max batches: {NUM_BENCHMARK_BATCHES}")
    
    # Benchmark single-threaded loader
    try:
        results_single = benchmark_loader(
            load_audio_train,
            num_batches=NUM_BENCHMARK_BATCHES,
            timeout_seconds=BENCHMARK_TIMEOUT,
            **config,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results_single = None
    
    # Benchmark grain loader
    try:
        results_grain = benchmark_loader(
            load_audio_train_grain,
            num_batches=NUM_BENCHMARK_BATCHES,
            timeout_seconds=BENCHMARK_TIMEOUT,
            **config,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results_grain = None
    
    # Compare results
    if results_single and results_grain:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Single-threaded: {results_single['samples_per_sec']:.2f} samples/sec")
        print(f"Grain loader:    {results_grain['samples_per_sec']:.2f} samples/sec")
        
        speedup = results_grain['samples_per_sec'] / results_single['samples_per_sec']
        if speedup > 1.0:
            print(f"Grain is {speedup:.2f}x faster")
        else:
            print(f"Grain is {1/speedup:.2f}x slower")
    
    # Bottleneck analysis
    identify_bottlenecks()
    
    # Try memory profiling if psutil is available
    print("\n" + "="*80)
    print("MEMORY PROFILING")
    print("="*80)
    profile_memory_and_io(timeout_seconds=MEMORY_PROFILE_TIMEOUT)


if __name__ == "__main__":
    main()

