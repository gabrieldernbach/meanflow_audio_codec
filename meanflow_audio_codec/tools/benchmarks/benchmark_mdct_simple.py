"""MDCT benchmark: JAX Metal vs NumPy baseline."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from meanflow_audio_codec.preprocessing.mdct import (imdct, imdct_baseline, mdct,
                                              mdct_baseline)


def benchmark(func, inputs, warmup=3, repeats=10):
    """Benchmark a function with warmup and averaging."""
    for _ in range(warmup):
        _ = func(*inputs)
    
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*inputs)
        jax.block_until_ready(result) if hasattr(result, 'block_until_ready') else None
        times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000


def main():
    """Main entry point for MDCT benchmark."""
    print("=" * 70)
    print("MDCT Benchmark: JAX Metal vs NumPy Baseline")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print()
    
    configs = [
        (2048, 512, "Small"),
        (16384, 2048, "Medium"),
        (65536, 2048, "Large"),
        (65536, 4096, "Large+BigWin"),
    ]
    
    results = []
    
    for signal_length, window_size, name in configs:
        print(f"{name}: {signal_length} samples, window={window_size}")
        print("-" * 70)
        
        key = jax.random.PRNGKey(42)
        x_jax = jax.random.normal(key, (signal_length,), dtype=jnp.float32)
        x_np = np.array(x_jax)
        hop_size = window_size // 2
        
        # NumPy baseline
        t_mean, t_std = benchmark(mdct_baseline, (x_np, window_size, hop_size))
        X_np = mdct_baseline(x_np, window_size, hop_size)
        t_mean2, t_std2 = benchmark(imdct_baseline, (X_np, window_size, hop_size))
        print(f"NumPy:     MDCT={t_mean:6.2f}±{t_std:.2f}ms, IMDCT={t_mean2:6.2f}±{t_std2:.2f}ms")
        
        # JAX Metal
        t_mean, t_std = benchmark(mdct, (x_jax, window_size, hop_size))
        X_jax = mdct(x_jax, window_size, hop_size)
        t_mean2, t_std2 = benchmark(imdct, (X_jax, window_size, hop_size))
        print(f"JAX Metal: MDCT={t_mean:6.2f}±{t_std:.2f}ms, IMDCT={t_mean2:6.2f}±{t_std2:.2f}ms")
        
        # Speedups vs baseline
        numpy_mdct_time = benchmark(mdct_baseline, (x_np, window_size, hop_size))[0]
        numpy_imdct_time = benchmark(imdct_baseline, (X_np, window_size, hop_size))[0]
        jax_mdct_time = benchmark(mdct, (x_jax, window_size, hop_size))[0]
        jax_imdct_time = benchmark(imdct, (X_jax, window_size, hop_size))[0]
        
        speedup_mdct = numpy_mdct_time / jax_mdct_time
        speedup_imdct = numpy_imdct_time / jax_imdct_time
        print(f"Speedup vs NumPy Baseline: MDCT={speedup_mdct:.2f}x, IMDCT={speedup_imdct:.2f}x")
        print()
        
        results.append({
            "name": name,
            "numpy_mdct": numpy_mdct_time,
            "numpy_imdct": numpy_imdct_time,
            "jax_mdct": jax_mdct_time,
            "jax_imdct": jax_imdct_time,
        })
    
    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE - MDCT Forward Transform")
    print("=" * 70)
    print(f"{'Config':<12} {'NumPy Baseline':<15} {'JAX Metal':<12} {'Speedup':<12}")
    print("-" * 70)
    for r in results:
        speedup_m = r["numpy_mdct"] / r["jax_mdct"]
        print(f"{r['name']:<12} {r['numpy_mdct']:>13.2f}ms "
              f"{r['jax_mdct']:>10.2f}ms {speedup_m:>10.2f}x")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - IMDCT Inverse Transform")
    print("=" * 70)
    print(f"{'Config':<12} {'NumPy Baseline':<15} {'JAX Metal':<12} {'Speedup':<12}")
    print("-" * 70)
    for r in results:
        speedup_i = r["numpy_imdct"] / r["jax_imdct"]
        print(f"{r['name']:<12} {r['numpy_imdct']:>13.2f}ms "
              f"{r['jax_imdct']:>10.2f}ms {speedup_i:>10.2f}x")


if __name__ == "__main__":
    main()

