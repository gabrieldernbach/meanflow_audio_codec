"""
Standalone benchmark for torch implementation that can be run in flowmo environment.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from flowmo
sys.path.insert(0, str(Path.home() / "git" / "flowmo"))
from dataloader import StreamingAudioDataset


def benchmark_torch_dataloader(
    data_dir: str,
    batch_size: int = 32,
    frame_sz: int = 256 * 256 * 3,
    buffer_size: int = 1024,
    num_workers: int = 0,
    num_batches: int = 100,
    warmup_batches: int = 10,
    seed: int = 42,
) -> dict:
    """Benchmark the torch flowmo dataloader."""
    print(f"\n{'='*70}")
    print("Benchmarking Torch Implementation (StreamingAudioDataset)")
    print(f"{'='*70}")
    
    # Find audio files
    files = [
        f for f in Path(data_dir).iterdir()
        if f.suffix.lower() == ".mp3" and f.is_file()
    ]
    
    if not files:
        print(f"  Error: No MP3 files found in {data_dir}")
        return None
    
    print(f"  Found {len(files)} MP3 files")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = StreamingAudioDataset(
        files=files,
        sr=44100,
        samples_per_frame=frame_sz,
        shuffle_buffer_size=buffer_size,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    
    # Warmup
    print(f"  Warming up with {warmup_batches} batches...")
    warmup_start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches - 1:
            break
    warmup_time = time.time() - warmup_start
    print(f"  Warmup completed in {warmup_time:.2f}s")
    
    # Actual timing
    print(f"  Processing {num_batches} batches for timing...")
    start_time = time.time()
    total_samples = 0
    batch_count = 0
    first_batch_time = None
    
    for batch in dataloader:
        if first_batch_time is None:
            first_batch_time = time.time() - start_time
        
        # batch shape is (batch_size, n_channels, frame_sz) for torch
        # Convert to samples: batch_size * frame_sz
        total_samples += batch.shape[0] * batch.shape[2]
        batch_count += 1
        
        if batch_count >= num_batches:
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    batches_per_sec = batch_count / elapsed if elapsed > 0 else 0
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
    
    results = {
        "elapsed_time": elapsed,
        "batches_processed": batch_count,
        "samples_processed": total_samples,
        "batches_per_second": batches_per_sec,
        "samples_per_second": samples_per_sec,
        "first_batch_time": first_batch_time,
    }
    
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  First batch time: {first_batch_time:.3f} seconds")
    print(f"  Batches processed: {batch_count}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Throughput: {batches_per_sec:.2f} batches/sec")
    print(f"  Throughput: {samples_per_sec:,.0f} samples/sec")
    
    return results


def main():
    """Run torch benchmark."""
    # Find data directory
    data_dir = os.environ.get("AUDIO_DATA_DIR", None)
    if data_dir is None:
        # Try common locations
        possible_dirs = [
            Path.home() / "datasets" / "wavegen",
            Path.home() / "git" / "flowmo" / "downloads",
        ]
        for dir_path in possible_dirs:
            if dir_path.exists():
                data_dir = str(dir_path)
                break
    
    if data_dir is None:
        print("Error: Could not find audio data directory.")
        print("Please set AUDIO_DATA_DIR environment variable")
        return
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Audio data directory does not exist: {data_dir}")
        return
    
    # Check for audio files
    files = [
        f for f in data_path.iterdir()
        if f.suffix.lower() == ".mp3" and f.is_file()
    ]
    if not files:
        print(f"Error: No MP3 files found in {data_dir}")
        return
    
    print(f"Found {len(files)} audio files in {data_dir}")
    
    # Benchmark parameters
    batch_size = 32
    frame_sz = 256 * 256 * 3  # 196608 samples
    buffer_size = 1000
    num_workers = 0  # Single-threaded for fair comparison
    num_batches = 50
    warmup_batches = 5
    seed = 42
    
    print("\n" + "=" * 70)
    print("Benchmark Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Number of files: {len(files)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Frame size: {frame_sz:,} samples")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Warmup batches: {warmup_batches}")
    print(f"  Timing batches: {num_batches}")
    print(f"  Seed: {seed}")
    print("=" * 70)
    
    # Benchmark torch implementation
    torch_results = benchmark_torch_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        frame_sz=frame_sz,
        buffer_size=buffer_size,
        num_workers=num_workers,
        num_batches=num_batches,
        warmup_batches=warmup_batches,
        seed=seed,
    )
    
    if torch_results:
        print("\n" + "=" * 70)
        print("Torch Results Summary:")
        print("=" * 70)
        print(f"  Batches/sec: {torch_results['batches_per_second']:.2f}")
        print(f"  Samples/sec: {torch_results['samples_per_second']:,.0f}")
        print(f"  First batch time: {torch_results['first_batch_time']:.3f}s")
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        import traceback
        print(f"Error running benchmark: {e}")
        traceback.print_exc()
        raise

