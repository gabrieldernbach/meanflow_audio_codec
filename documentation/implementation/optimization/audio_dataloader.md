# Audio Dataloader Optimization Opportunities

## Current Performance

Based on benchmark results:
- **Throughput**: ~74M samples/sec
- **Batches/sec**: ~11.84
- Already ~1.8x faster than PyTorch implementation

## Analysis of Current Implementation

### Bottlenecks (in order of impact)

1. **MP3 Decoding** (largest) - Already optimized with minimp3py C library
2. **File I/O** - Already optimized with thread-based prefetching
3. **Array Operations** - NumPy operations (padding, transpose, reshape, stack)
4. **NumPy→JAX Conversion** - `jnp.asarray()` in training loop copies data

### Current Code Structure

```python
# NumPy arrays returned from dataloader
batch = np.ndarray  # shape: (batch_size, frame_sz, n_channels)

# Converted to JAX in training loop
x = jnp.asarray(batch)  # Creates copy
```

## Optimization Opportunities

### 1. **Use JAX Arrays Directly** ⭐ (Medium-High Impact)

**Current**: Datasloader returns NumPy arrays, training loop converts with `jnp.asarray()` (creates copy)

**Optimization**: Return JAX arrays directly from dataloader

**Benefits**:
- Eliminates copy overhead in training loop
- Better integration with JAX ecosystem
- Enables future JIT compilation opportunities

**Implementation Notes**:
- Use `jnp.array()` or `jnp.asarray()` when creating arrays
- Requires ensuring arrays are on correct device (CPU for dataloader)
- May need to handle device placement explicitly

**Estimated Impact**: 5-15% speedup (eliminates one copy per batch)

---

### 2. **Replace NumPy Random with JAX Random** ⭐ (Medium Impact)

**Current**: Uses `np.random.Generator` and `np.random.default_rng()`

**Optimization**: Use JAX random number generation

**Benefits**:
- Enables JIT compilation of random operations
- More consistent with JAX ecosystem
- Better performance for random number generation

**Challenges**:
- JAX random requires PRNGKey (stateless), not stateful generators
- Need to thread keys through the pipeline
- More complex state management

**Estimated Impact**: 5-10% speedup (better RNG performance + JIT potential)

**Example**:
```python
# Current
rng = np.random.default_rng(seed)
n_prepend = rng.integers(0, frame_sz + 1)

# Proposed
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
n_prepend = jax.random.randint(subkey, (), 0, frame_sz + 1)
```

---

### 3. **Pre-allocate Batch Buffer** (Low-Medium Impact)

**Current**: `batch()` function uses list.append() then `np.stack()` each time

**Optimization**: Pre-allocate buffer array and fill it

**Benefits**:
- Reduces memory allocations
- Faster than repeated list.append() + stack()

**Code Location**: `batch()` function (lines 186-202)

**Estimated Impact**: 2-5% speedup

---

### 4. **JIT Compile Frame Processing** ⚠️ (Limited Impact)

**Current**: NumPy operations for padding and reshaping

**Potential**: Use `@jax.jit` on `_prepend_and_pad_audio` and frame reshaping

**Challenges**:
- Requires JAX arrays (see optimization #1)
- Requires JAX random (see optimization #2)
- JIT overhead may not be worth it for simple operations
- Iterator-based pipeline makes JIT difficult

**Estimated Impact**: 0-5% speedup (JIT overhead may negate benefits for simple ops)

**Verdict**: **Not Recommended** - NumPy operations are already optimized, JIT overhead likely exceeds benefits

---

### 5. **Optimize Padding Operations** (Low Impact)

**Current**: Uses `np.pad()` which may allocate new array

**Optimization**: Use in-place operations or pre-allocate when possible

**Code Location**: `_prepend_and_pad_audio()` (lines 265-278)

**Estimated Impact**: 1-3% speedup

---

### 6. **Cython** ❌ (Not Recommended)

**Why Not**:
- MP3 decoding already in C (minimp3py)
- NumPy operations already optimized C code
- Python overhead is minimal (iterator protocol, function calls)
- Cython would add complexity with minimal benefit

**Estimated Impact**: Negligible (0-2%)

---

## Recommended Priority

1. **High Priority**: Use JAX arrays directly (#1)
   - Clear benefit, relatively straightforward
   - Eliminates copy overhead

2. **Medium Priority**: Replace NumPy random with JAX random (#2)
   - Good for JAX ecosystem consistency
   - Enables future optimizations
   - Moderate complexity increase

3. **Low Priority**: Pre-allocate batch buffer (#3)
   - Easy to implement
   - Small but measurable benefit

4. **Not Recommended**: JIT compilation (#4) and Cython (#6)
   - Limited or no benefit for current use case

## Implementation Considerations

### Breaking Changes

If implementing #1 (JAX arrays):
- May require updating all code that uses the dataloader
- Need to handle device placement (CPU vs GPU)
- JAX arrays behave slightly differently than NumPy in some edge cases

### Testing

All optimizations should:
1. Maintain API compatibility (or document breaking changes)
2. Pass existing tests
3. Show measurable performance improvement in benchmarks

## Measurement

To validate optimizations, run:
```bash
uv run benchmarks/benchmark_audio_simple_test.py
uv run benchmark_audio_vs_torch.py
```

Monitor:
- Batches/sec
- Samples/sec
- First batch time
- Memory usage (may change with JAX arrays)

