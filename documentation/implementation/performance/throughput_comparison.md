# Audio Dataloader Throughput Comparison

## Benchmark Results

### Test Configuration
- **Data directory**: `/Users/gabrieldernbach/datasets/wavegen`
- **Number of files**: 458 MP3 files
- **Batch size**: 32
- **Frame size**: 196,608 samples (256 × 256 × 3)
- **Buffer size**: 1000
- **Prefetch (JAX only)**: 4
- **Warmup batches**: 5
- **Timing batches**: 50
- **Seed**: 42
- **Num workers (Torch)**: 0 (single-threaded for fair comparison)

## Results

### JAX Implementation (`build_audio_pipeline`)

**Location**: `meanflow_audio_codec/datasets/audio.py`

- **Batches/sec**: 11.84
- **Samples/sec**: 74,471,547
- **First batch time**: 0.273s
- **Warmup time**: 3.05s
- **Total time (50 batches)**: 4.22s

**Key Features**:
- Uses `minimp3py` for fast MP3 loading
- Thread-based prefetching (4 files ahead)
- Composable pipeline architecture
- Streaming shuffle buffer

### Torch Implementation (`StreamingAudioDataset`)

**Location**: `git/flowmo/dataloader.py`

- **Batches/sec**: 6.57
- **Samples/sec**: 41,317,700
- **First batch time**: 3.591s
- **Warmup time**: 8.07s
- **Total time (50 batches)**: 7.61s

**Key Features**:
- Uses `librosa` for MP3 loading
- PyTorch DataLoader integration
- Supports multi-worker loading (not tested here)
- Streaming shuffle buffer

## Performance Comparison

| Metric | JAX | Torch | Ratio (JAX/Torch) |
|--------|-----|-------|-------------------|
| **Batches/sec** | 11.84 | 6.57 | **1.80x faster** |
| **Samples/sec** | 74,471,547 | 41,317,700 | **1.80x faster** |
| **Time (50 batches)** | 4.22s | 7.61s | **1.80x faster** |
| **First batch time** | 0.273s | 3.591s | **13.16x faster** |
| **Warmup time** | 3.05s | 8.07s | **2.65x faster** |

## Analysis

### Why JAX Implementation is Faster

1. **MP3 Loading Library**:
   - JAX uses `minimp3py` (C-based, optimized)
   - Torch uses `librosa` (Python-based, slower)
   - This is the primary bottleneck difference

2. **Prefetching**:
   - JAX has thread-based prefetching (4 files ahead)
   - Torch relies on DataLoader prefetching (not used with num_workers=0)

3. **First Batch Latency**:
   - JAX: 0.273s (very fast startup)
   - Torch: 3.591s (slower due to librosa initialization and first file load)

4. **Overall Throughput**:
   - JAX processes ~1.8x more batches per second
   - JAX processes ~1.8x more samples per second

### Notes

- Both implementations use similar shuffle buffer strategies
- Both are single-threaded in this comparison (fair comparison)
- Torch implementation supports multi-worker loading which could improve throughput
- JAX implementation uses minimp3py which is significantly faster than librosa for MP3 loading

## Conclusion

The JAX implementation (`audio.py`) demonstrates **1.8x better throughput** than the Torch implementation, primarily due to:
1. Faster MP3 loading with `minimp3py` vs `librosa`
2. Thread-based prefetching
3. Lower first-batch latency

The JAX implementation is particularly advantageous for:
- Fast iteration during development (low first-batch latency)
- High-throughput training scenarios
- Single-threaded workloads

