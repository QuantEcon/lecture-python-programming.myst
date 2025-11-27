---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# JAX Performance Benchmark - Jupyter Book Execution

This file tests JAX performance when executed through Jupyter Book's notebook execution.
Compare results with direct script and nbconvert execution.

```{code-cell} ipython3
import time
import platform
import os
import json
from datetime import datetime

# Global results dictionary for JSON output
RESULTS = {
    "timestamp": datetime.now().isoformat(),
    "execution_method": "jupyter_book",
    "system": {},
    "benchmarks": {}
}

print("=" * 60)
print("JUPYTER BOOK EXECUTION BENCHMARK")
print("=" * 60)
print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"CPU Count: {os.cpu_count()}")

RESULTS["system"]["platform"] = platform.platform()
RESULTS["system"]["python_version"] = platform.python_version()
RESULTS["system"]["cpu_count"] = os.cpu_count()
```

```{code-cell} ipython3
# Import JAX and check devices
import jax
import jax.numpy as jnp

devices = jax.devices()
default_backend = jax.default_backend()
has_gpu = any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices)

print(f"JAX devices: {devices}")
print(f"Default backend: {default_backend}")
print(f"GPU Available: {has_gpu}")

RESULTS["system"]["jax_devices"] = [str(d) for d in devices]
RESULTS["system"]["jax_backend"] = default_backend
RESULTS["system"]["has_gpu"] = has_gpu
```

```{code-cell} ipython3
# Define JIT-compiled function
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

print("matmul function defined with @jax.jit")
```

```{code-cell} ipython3
# Benchmark 1: Small matrix (1000x1000) - includes JIT compilation
print("\n" + "=" * 60)
print("BENCHMARK 1: Small Matrix (1000x1000)")
print("=" * 60)

n = 1000
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (n, n))
B = jax.random.normal(key, (n, n))

# Warm-up run (includes compilation)
start = time.perf_counter()
C = matmul(A, B).block_until_ready()
warmup_time = time.perf_counter() - start
print(f"Warm-up (includes JIT compile): {warmup_time:.3f} seconds")

# Compiled run
start = time.perf_counter()
C = matmul(A, B).block_until_ready()
compiled_time = time.perf_counter() - start
print(f"Compiled execution: {compiled_time:.3f} seconds")

RESULTS["benchmarks"]["matmul_1000x1000"] = {
    "warmup": warmup_time,
    "compiled": compiled_time
}
```

```{code-cell} ipython3
# Benchmark 2: Large matrix (3000x3000) - triggers recompilation
print("\n" + "=" * 60)
print("BENCHMARK 2: Large Matrix (3000x3000)")
print("=" * 60)

n = 3000
A = jax.random.normal(key, (n, n))
B = jax.random.normal(key, (n, n))

# Warm-up run (recompilation for new size)
start = time.perf_counter()
C = matmul(A, B).block_until_ready()
warmup_time = time.perf_counter() - start
print(f"Warm-up (recompile for new size): {warmup_time:.3f} seconds")

# Compiled run
start = time.perf_counter()
C = matmul(A, B).block_until_ready()
compiled_time = time.perf_counter() - start
print(f"Compiled execution: {compiled_time:.3f} seconds")

RESULTS["benchmarks"]["matmul_3000x3000"] = {
    "warmup": warmup_time,
    "compiled": compiled_time
}
```

```{code-cell} ipython3
# Benchmark 3: Element-wise operations (50M elements)
print("\n" + "=" * 60)
print("BENCHMARK 3: Element-wise Operations (50M elements)")
print("=" * 60)

@jax.jit
def elementwise_ops(x):
    return jnp.cos(x**2) + jnp.sin(x)

x = jax.random.normal(key, (50_000_000,))

# Warm-up
start = time.perf_counter()
y = elementwise_ops(x).block_until_ready()
warmup_time = time.perf_counter() - start
print(f"Warm-up (includes JIT compile): {warmup_time:.3f} seconds")

# Compiled
start = time.perf_counter()
y = elementwise_ops(x).block_until_ready()
compiled_time = time.perf_counter() - start
print(f"Compiled execution: {compiled_time:.3f} seconds")

RESULTS["benchmarks"]["elementwise_50M"] = {
    "warmup": warmup_time,
    "compiled": compiled_time
}
```

```{code-cell} ipython3
# Benchmark 4: Multiple small operations (simulates lecture cells)
print("\n" + "=" * 60)
print("BENCHMARK 4: Multiple Small Operations (lecture simulation)")
print("=" * 60)

total_start = time.perf_counter()
multi_op_results = {}

# Simulate multiple cell executions with different operations
for i, size in enumerate([100, 500, 1000, 2000, 3000]):
    @jax.jit
    def compute(a, b):
        return jnp.dot(a, b) + jnp.sum(a)
    
    A = jax.random.normal(key, (size, size))
    B = jax.random.normal(key, (size, size))
    
    start = time.perf_counter()
    result = compute(A, B).block_until_ready()
    elapsed = time.perf_counter() - start
    print(f"  Size {size}x{size}: {elapsed:.3f} seconds")
    multi_op_results[f"size_{size}x{size}"] = elapsed

total_time = time.perf_counter() - total_start
print(f"\nTotal time for all operations: {total_time:.3f} seconds")

multi_op_results["total_time"] = total_time
RESULTS["benchmarks"]["multi_operations"] = multi_op_results
```

```{code-cell} ipython3
# Save results to JSON file
output_file = "benchmark_results_jupyterbook.json"
with open(output_file, 'w') as f:
    json.dump(RESULTS, f, indent=2)

print("\n" + "=" * 60)
print("JUPYTER BOOK EXECUTION BENCHMARK COMPLETE")
print("=" * 60)
print(f"\nResults saved to {output_file}")
```
