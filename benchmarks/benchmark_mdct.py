"""Benchmark MDCT implementations: JAX Metal vs CPU vs NumPy baseline."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from meanflow_audio_codec.preprocessing.mdct import (
    mdct, imdct, mdct_fft, imdct_fft, mdct_baseline, imdct_baseline
)


def benchmark(func, inputs, warmup=3, repeats=10):
    """Benchmark a function with warmup and averaging."""
    # Warmup
    for _ in range(warmup):
        _ = func(*inputs)
    
    # Time it
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*inputs)
        jax.block_until_ready(result) if hasattr(result, 'block_until_ready') else None
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times), result


def run_benchmarks():
    """Run comprehensive MDCT benchmarks."""
    print("=" * 70)
    print("MDCT Implementation Benchmarks")
    print("=" * 70)
    print()
    
    # Test configurations: (signal_length, window_size, description)
    configs = [
        (2048, 512, "Small signal"),
        (16384, 2048, "Medium signal"),
        (65536, 2048, "Large signal"),
        (65536, 4096, "Large signal, large window"),
    ]
    
    results = []
    
    for signal_length, window_size, desc in configs:
        print(f"\n{desc}: signal_length={signal_length}, window_size={window_size}")
        print("-" * 70)
        
        # Generate test data
        key = jax.random.PRNGKey(42)
        x_jax = jax.random.normal(key, (signal_length,), dtype=jnp.float32)
        x_np = np.array(x_jax)
        hop_size = window_size // 2
        
        # Skip FFT benchmarks for small windows (will use direct anyway)
        use_fft = window_size >= 512
        
        config_results = {"config": desc, "signal_length": signal_length, "window_size": window_size}
        
        # 1. NumPy Baseline (CPU)
        print("NumPy baseline (CPU)...")
        try:
            t_mean, t_std, X_baseline = benchmark(mdct_baseline, (x_np, window_size, hop_size))
            _, _, xr_baseline = benchmark(imdct_baseline, (X_baseline, window_size, hop_size))
            print(f"  MDCT:  {t_mean*1000:.2f} ± {t_std*1000:.2f} ms")
            t_mean2, t_std2, _ = benchmark(imdct_baseline, (X_baseline, window_size, hop_size))
            print(f"  IMDCT: {t_mean2*1000:.2f} ± {t_std2*1000:.2f} ms")
            config_results["numpy_mdct"] = t_mean * 1000
            config_results["numpy_imdct"] = t_mean2 * 1000
        except Exception as e:
            print(f"  Error: {e}")
            config_results["numpy_mdct"] = None
            config_results["numpy_imdct"] = None
        
        # 2. JAX Direct (Metal)
        print("JAX direct (Metal)...")
        try:
            metal_devices = [d for d in jax.devices() if d.platform == "METAL"]
            if metal_devices:
                with jax.default_device(metal_devices[0]):
                    x_metal = jax.device_put(x_jax, metal_devices[0])
                    t_mean, t_std, X_metal = benchmark(mdct, (x_metal, window_size, hop_size))
                    print(f"  MDCT:  {t_mean*1000:.2f} ± {t_std*1000:.2f} ms")
                    t_mean2, t_std2, _ = benchmark(imdct, (X_metal, window_size, hop_size))
                    print(f"  IMDCT: {t_mean2*1000:.2f} ± {t_std2*1000:.2f} ms")
                    config_results["jax_metal_mdct"] = t_mean * 1000
                    config_results["jax_metal_imdct"] = t_mean2 * 1000
            else:
                print("  Metal not available")
                config_results["jax_metal_mdct"] = None
                config_results["jax_metal_imdct"] = None
        except Exception as e:
            print(f"  Error: {e}")
            config_results["jax_metal_mdct"] = None
            config_results["jax_metal_imdct"] = None
        
        # 3. JAX Direct (CPU) - compile explicitly for CPU
        print("JAX direct (CPU)...")
        try:
            # Force CPU compilation
            with jax.default_backend("cpu"):
                # JIT compile functions for CPU
                mdct_cpu = jax.jit(mdct, backend="cpu")
                imdct_cpu = jax.jit(imdct, backend="cpu")
                x_cpu = jnp.array(x_np)  # Use numpy array, JAX will handle on CPU
                t_mean, t_std, X_cpu = benchmark(mdct_cpu, (x_cpu, window_size, hop_size))
                print(f"  MDCT:  {t_mean*1000:.2f} ± {t_std*1000:.2f} ms")
                t_mean2, t_std2, _ = benchmark(imdct_cpu, (X_cpu, window_size, hop_size))
                print(f"  IMDCT: {t_mean2*1000:.2f} ± {t_std2*1000:.2f} ms")
                config_results["jax_cpu_mdct"] = t_mean * 1000
                config_results["jax_cpu_imdct"] = t_mean2 * 1000
        except Exception as e:
            print(f"  Error: {e}")
            config_results["jax_cpu_mdct"] = None
            config_results["jax_cpu_imdct"] = None
        
        # 4. JAX FFT (Metal) - only for large windows
        if use_fft:
            print("JAX FFT (Metal)...")
            try:
                metal_devices = [d for d in jax.devices() if d.platform == "METAL"]
                if metal_devices:
                    with jax.default_device(metal_devices[0]):
                        x_metal = jax.device_put(x_jax, metal_devices[0])
                        t_mean, t_std, X_metal_fft = benchmark(mdct_fft, (x_metal, window_size, hop_size))
                        print(f"  MDCT:  {t_mean*1000:.2f} ± {t_std*1000:.2f} ms")
                        t_mean2, t_std2, _ = benchmark(imdct_fft, (X_metal_fft, window_size, hop_size))
                        print(f"  IMDCT: {t_mean2*1000:.2f} ± {t_std2*1000:.2f} ms")
                        config_results["jax_metal_fft_mdct"] = t_mean * 1000
                        config_results["jax_metal_fft_imdct"] = t_mean2 * 1000
                else:
                    print("  Metal not available")
                    config_results["jax_metal_fft_mdct"] = None
                    config_results["jax_metal_fft_imdct"] = None
            except Exception as e:
                print(f"  Error: {e} (Metal FFT may not be supported)")
                config_results["jax_metal_fft_mdct"] = None
                config_results["jax_metal_fft_imdct"] = None
            
            # 5. JAX FFT (CPU)
            print("JAX FFT (CPU)...")
            try:
                with jax.default_backend("cpu"):
                    mdct_fft_cpu = jax.jit(mdct_fft, backend="cpu")
                    imdct_fft_cpu = jax.jit(imdct_fft, backend="cpu")
                    x_cpu = jnp.array(x_np)
                    t_mean, t_std, X_cpu_fft = benchmark(mdct_fft_cpu, (x_cpu, window_size, hop_size))
                    print(f"  MDCT:  {t_mean*1000:.2f} ± {t_std*1000:.2f} ms")
                    t_mean2, t_std2, _ = benchmark(imdct_fft_cpu, (X_cpu_fft, window_size, hop_size))
                    print(f"  IMDCT: {t_mean2*1000:.2f} ± {t_std2*1000:.2f} ms")
                    config_results["jax_cpu_fft_mdct"] = t_mean * 1000
                    config_results["jax_cpu_fft_imdct"] = t_mean2 * 1000
            except Exception as e:
                print(f"  Error: {e}")
                config_results["jax_cpu_fft_mdct"] = None
                config_results["jax_cpu_fft_imdct"] = None
        
        results.append(config_results)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY - MDCT Forward Transform (ms)")
    print("=" * 70)
    print(f"{'Config':<30} {'NumPy':<10} {'JAX Metal':<12} {'JAX CPU':<12} {'JAX Metal FFT':<15} {'JAX CPU FFT':<12}")
    print("-" * 70)
    
    for r in results:
        config = r["config"]
        np_t = f"{r['numpy_mdct']:.2f}" if r['numpy_mdct'] else "N/A"
        metal_t = f"{r['jax_metal_mdct']:.2f}" if r['jax_metal_mdct'] else "N/A"
        cpu_t = f"{r['jax_cpu_mdct']:.2f}" if r['jax_cpu_mdct'] else "N/A"
        metal_fft_t = f"{r.get('jax_metal_fft_mdct', None):.2f}" if r.get('jax_metal_fft_mdct') else "N/A"
        cpu_fft_t = f"{r.get('jax_cpu_fft_mdct', None):.2f}" if r.get('jax_cpu_fft_mdct') else "N/A"
        print(f"{config:<30} {np_t:<10} {metal_t:<12} {cpu_t:<12} {metal_fft_t:<15} {cpu_fft_t:<12}")
    
    print("\n" + "=" * 70)
    print("SUMMARY - IMDCT Inverse Transform (ms)")
    print("=" * 70)
    print(f"{'Config':<30} {'NumPy':<10} {'JAX Metal':<12} {'JAX CPU':<12} {'JAX Metal FFT':<15} {'JAX CPU FFT':<12}")
    print("-" * 70)
    
    for r in results:
        config = r["config"]
        np_t = f"{r['numpy_imdct']:.2f}" if r['numpy_imdct'] else "N/A"
        metal_t = f"{r['jax_metal_imdct']:.2f}" if r['jax_metal_imdct'] else "N/A"
        cpu_t = f"{r['jax_cpu_imdct']:.2f}" if r['jax_cpu_imdct'] else "N/A"
        metal_fft_t = f"{r.get('jax_metal_fft_imdct', None):.2f}" if r.get('jax_metal_fft_imdct') else "N/A"
        cpu_fft_t = f"{r.get('jax_cpu_fft_imdct', None):.2f}" if r.get('jax_cpu_fft_imdct') else "N/A"
        print(f"{config:<30} {np_t:<10} {metal_t:<12} {cpu_t:<12} {metal_fft_t:<15} {cpu_fft_t:<12}")
    
    # Speedup calculations
    print("\n" + "=" * 70)
    print("SPEEDUP vs NumPy Baseline")
    print("=" * 70)
    
    for r in results:
        if r['numpy_mdct']:
            print(f"\n{r['config']}:")
            if r['jax_metal_mdct']:
                speedup = r['numpy_mdct'] / r['jax_metal_mdct']
                print(f"  JAX Metal direct: {speedup:.2f}x")
            if r.get('jax_metal_fft_mdct'):
                speedup = r['numpy_mdct'] / r['jax_metal_fft_mdct']
                print(f"  JAX Metal FFT:    {speedup:.2f}x")
            if r['jax_cpu_mdct']:
                speedup = r['numpy_mdct'] / r['jax_cpu_mdct']
                print(f"  JAX CPU direct:   {speedup:.2f}x")
            if r.get('jax_cpu_fft_mdct'):
                speedup = r['numpy_mdct'] / r['jax_cpu_fft_mdct']
                print(f"  JAX CPU FFT:      {speedup:.2f}x")


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    print()
    run_benchmarks()

