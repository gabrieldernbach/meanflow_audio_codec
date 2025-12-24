"""Performance metrics for training and inference.

This module provides utilities for tracking training time, inference time,
memory usage, and parameter counting.
"""

import time
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_reduce

try:
    import psutil
except ImportError:
    psutil = None


class TrainingTimer:
    """Context manager for tracking training time.
    
    Usage:
        timer = TrainingTimer()
        with timer:
            # training code
        elapsed = timer.elapsed()
    """
    
    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        return False
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


def inference_time(
    fn: Any,
    *args,
    num_warmup: int = 10,
    num_runs: int = 100,
    **kwargs,
) -> dict[str, float]:
    """Measure inference time for a function.
    
    Performs warmup runs to JIT-compile if needed, then measures average
    inference time over multiple runs.
    
    Args:
        fn: Function to time (can be JIT-compiled)
        *args: Positional arguments to pass to fn
        num_warmup: Number of warmup runs (default 10)
        num_runs: Number of timing runs (default 100)
        **kwargs: Keyword arguments to pass to fn
    
    Returns:
        Dictionary with:
            - "mean": Mean inference time in seconds
            - "std": Standard deviation in seconds
            - "min": Minimum time in seconds
            - "max": Maximum time in seconds
            - "total": Total time for all runs in seconds
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = fn(*args, **kwargs)
    
    # Synchronize before timing (important for GPU)
    if hasattr(jax.devices()[0], "synchronize"):
        jax.devices()[0].synchronize()
    
    # Timing runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        # Synchronize after computation
        if hasattr(jax.devices()[0], "synchronize"):
            jax.devices()[0].synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    times_array = np.array(times)
    return {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "total": float(np.sum(times_array)),
    }


def memory_usage() -> dict[str, Any]:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with:
            - "gpu_memory_used_mb": GPU memory used in MB (if available)
            - "gpu_memory_total_mb": Total GPU memory in MB (if available)
            - "cpu_memory_used_mb": CPU memory used in MB (if psutil available)
            - "cpu_memory_total_mb": Total CPU memory in MB (if psutil available)
            - "cpu_memory_percent": CPU memory usage percentage (if psutil available)
    """
    result: dict[str, Any] = {}
    
    # GPU memory (JAX)
    try:
        devices = jax.devices()
        if devices:
            device = devices[0]
            if hasattr(device, "memory_stats"):
                stats = device.memory_stats()
                # Memory stats format varies by backend
                # For Metal: may have different keys
                # For CUDA: typically has 'bytes_in_use' and 'bytes_limit'
                if "bytes_in_use" in stats:
                    result["gpu_memory_used_mb"] = stats["bytes_in_use"] / (1024 ** 2)
                if "bytes_limit" in stats:
                    result["gpu_memory_total_mb"] = stats["bytes_limit"] / (1024 ** 2)
                elif "bytes_reserved" in stats:
                    # Fallback for some backends
                    result["gpu_memory_total_mb"] = stats["bytes_reserved"] / (1024 ** 2)
    except Exception:
        # Backend may not support memory stats
        pass
    
    # CPU memory (psutil)
    if psutil is not None:
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            result["cpu_memory_used_mb"] = mem_info.rss / (1024 ** 2)
            
            # System memory
            sys_mem = psutil.virtual_memory()
            result["cpu_memory_total_mb"] = sys_mem.total / (1024 ** 2)
            result["cpu_memory_percent"] = sys_mem.percent
        except Exception:
            pass
    
    return result


def count_parameters(params: Any) -> dict[str, Any]:
    """Count parameters in a model.
    
    Args:
        params: Model parameters (Flax params dict or JAX pytree)
    
    Returns:
        Dictionary with:
            - "total": Total number of parameters
            - "total_millions": Total parameters in millions
            - "by_module": Dictionary mapping module names to parameter counts
            - "trainable": Number of trainable parameters (same as total for now)
    """
    # Count parameters per module
    by_module: dict[str, int] = {}
    
    def count_node(path: tuple, node: Any) -> int:
        """Count parameters in a single node."""
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            count = int(node.size)
            # Build module path string
            if path:
                module_path = "/".join(str(p) for p in path)
                by_module[module_path] = by_module.get(module_path, 0) + count
            return count
        return 0
    
    # Traverse parameter tree
    total = tree_reduce(
        lambda a, b: a + b,
        tree_map(count_node, params),
        0,
    )
    
    return {
        "total": int(total),
        "total_millions": float(total / 1e6),
        "by_module": by_module,
        "trainable": int(total),  # All parameters are trainable in Flax
    }


@contextmanager
def memory_profiler():
    """Context manager for profiling memory usage.
    
    Usage:
        with memory_profiler() as profiler:
            # code to profile
        before = profiler["before"]
        after = profiler["after"]
        delta = profiler["delta"]
    """
    before = memory_usage()
    profiler = {"before": before}
    
    try:
        yield profiler
    finally:
        after = memory_usage()
        profiler["after"] = after
        
        # Compute delta
        delta: dict[str, Any] = {}
        for key in set(before.keys()) | set(after.keys()):
            if key.endswith("_mb") or key.endswith("_percent"):
                before_val = before.get(key, 0)
                after_val = after.get(key, 0)
                delta[key.replace("_mb", "_delta_mb").replace("_percent", "_delta_percent")] = (
                    after_val - before_val
                )
        profiler["delta"] = delta

