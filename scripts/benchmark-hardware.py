"""
"""Hardware benchmark script for CI runners.
Compares CPU and GPU performance to diagnose slowdowns.
Works on both CPU-only (GitHub Actions) and GPU (RunsOn) runners.
"""
import time
import platform
import os
import json
from datetime import datetime

# Global results dictionary for JSON output
RESULTS = {
    "timestamp": datetime.now().isoformat(),
    "execution_method": "bare_metal",
    "system": {},
    "benchmarks": {}
}

def get_cpu_info():
    """Get CPU information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    
    RESULTS["system"]["platform"] = platform.platform()
    RESULTS["system"]["processor"] = platform.processor()
    RESULTS["system"]["python_version"] = platform.python_version()
    
    # Try to get CPU frequency
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    print(f"CPU Model: {line.split(':')[1].strip()}")
                    break
    except:
        pass
    
    # Try to get CPU frequency
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'cpu MHz' in line:
                    print(f"CPU MHz: {line.split(':')[1].strip()}")
                    break
    except:
        pass
    
    # CPU count
    print(f"CPU Count: {os.cpu_count()}")
    RESULTS["system"]["cpu_count"] = os.cpu_count()
    
    # Check for GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"GPU: {gpu_info}")
            RESULTS["system"]["gpu"] = gpu_info
        else:
            print("GPU: None detected")
            RESULTS["system"]["gpu"] = None
    except:
        print("GPU: None detected (nvidia-smi not available)")
        RESULTS["system"]["gpu"] = None
    
    print()

def benchmark_cpu_pure_python():
    """Pure Python CPU benchmark."""
    print("=" * 60)
    print("CPU BENCHMARK: Pure Python")
    print("=" * 60)
    
    results = {}
    
    # Integer computation
    start = time.perf_counter()
    total = sum(i * i for i in range(10_000_000))
    elapsed = time.perf_counter() - start
    print(f"Integer sum (10M iterations): {elapsed:.3f} seconds")
    results["integer_sum_10M"] = elapsed
    
    # Float computation
    start = time.perf_counter()
    total = 0.0
    for i in range(1_000_000):
        total += (i * 0.1) ** 0.5
    elapsed = time.perf_counter() - start
    print(f"Float sqrt (1M iterations): {elapsed:.3f} seconds")
    results["float_sqrt_1M"] = elapsed
    
    RESULTS["benchmarks"]["pure_python"] = results
    print()

def benchmark_cpu_numpy():
    """NumPy CPU benchmark."""
    import numpy as np
    
    print("=" * 60)
    print("CPU BENCHMARK: NumPy")
    print("=" * 60)
    
    results = {}
    
    # Matrix multiplication
    n = 3000
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    start = time.perf_counter()
    C = A @ B
    elapsed = time.perf_counter() - start
    print(f"Matrix multiply ({n}x{n}): {elapsed:.3f} seconds")
    results["matmul_3000x3000"] = elapsed
    
    # Element-wise operations
    x = np.random.randn(50_000_000)
    
    start = time.perf_counter()
    y = np.cos(x**2) + np.sin(x)
    elapsed = time.perf_counter() - start
    print(f"Element-wise ops (50M elements): {elapsed:.3f} seconds")
    results["elementwise_50M"] = elapsed
    
    RESULTS["benchmarks"]["numpy"] = results
    print()

def benchmark_gpu_jax():
    """JAX benchmark (GPU if available, otherwise CPU)."""
    try:
        import jax
        import jax.numpy as jnp
        
        results = {}
        
        devices = jax.devices()
        default_backend = jax.default_backend()
        
        # Check if GPU is available
        has_gpu = any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices)
        
        results["has_gpu"] = has_gpu
        results["default_backend"] = default_backend
        results["devices"] = [str(d) for d in devices]
        
        print("=" * 60)
        if has_gpu:
            print("JAX BENCHMARK: GPU")
        else:
            print("JAX BENCHMARK: CPU (no GPU detected)")
        print("=" * 60)
        
        print(f"JAX devices: {devices}")
        print(f"Default backend: {default_backend}")
        print(f"GPU Available: {has_gpu}")
        print()
        
        # Warm-up JIT compilation
        print("Warming up JIT compilation...")
        n = 1000
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (n, n))
        B = jax.random.normal(key, (n, n))
        
        @jax.jit
        def matmul(a, b):
            return jnp.dot(a, b)
        
        # Warm-up run (includes compilation)
        start = time.perf_counter()
        C = matmul(A, B).block_until_ready()
        warmup_time = time.perf_counter() - start
        print(f"Warm-up (includes JIT compile, {n}x{n}): {warmup_time:.3f} seconds")
        results["matmul_1000x1000_warmup"] = warmup_time
        
        # Actual benchmark (compiled)
        start = time.perf_counter()
        C = matmul(A, B).block_until_ready()
        elapsed = time.perf_counter() - start
        print(f"Matrix multiply compiled ({n}x{n}): {elapsed:.3f} seconds")
        results["matmul_1000x1000_compiled"] = elapsed
        
        # Larger matrix
        n = 3000
        A = jax.random.normal(key, (n, n))
        B = jax.random.normal(key, (n, n))
        
        # Warm-up for new size
        start = time.perf_counter()
        C = matmul(A, B).block_until_ready()
        warmup_time = time.perf_counter() - start
        print(f"Warm-up (recompile for {n}x{n}): {warmup_time:.3f} seconds")
        results["matmul_3000x3000_warmup"] = warmup_time
        
        # Benchmark compiled
        start = time.perf_counter()
        C = matmul(A, B).block_until_ready()
        elapsed = time.perf_counter() - start
        print(f"Matrix multiply compiled ({n}x{n}): {elapsed:.3f} seconds")
        results["matmul_3000x3000_compiled"] = elapsed)
        
        # Element-wise GPU benchmark
        x = jax.random.normal(key, (50_000_000,))
        
        @jax.jit
        def elementwise_ops(x):
            return jnp.cos(x**2) + jnp.sin(x)
        
        # Warm-up
        start = time.perf_counter()
        y = elementwise_ops(x).block_until_ready()
        warmup_time = time.perf_counter() - start
        print(f"Element-wise warm-up (50M): {warmup_time:.3f} seconds")
        results["elementwise_50M_warmup"] = warmup_time
        
        # Compiled
        start = time.perf_counter()
        y = elementwise_ops(x).block_until_ready()
        elapsed = time.perf_counter() - start
        print(f"Element-wise compiled (50M): {elapsed:.3f} seconds")
        results["elementwise_50M_compiled"] = elapsed
        
        RESULTS["benchmarks"]["jax"] = results
        print()
        
    except ImportError as e:
        print(f"JAX not available: {e}")
        RESULTS["benchmarks"]["jax"] = {"error": str(e)}
    except Exception as e:
        print(f"JAX benchmark failed: {e}")
        RESULTS["benchmarks"]["jax"] = {"error": str(e)})

def benchmark_numba():
    """Numba CPU benchmark."""
    try:
        import numba
        import numpy as np
        
        results = {}
        
        print("=" * 60)
        print("CPU BENCHMARK: Numba")
        print("=" * 60)
        
        @numba.jit(nopython=True)
        def numba_sum(n):
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        # Warm-up (compilation)
        start = time.perf_counter()
        result = numba_sum(10_000_000)
        warmup_time = time.perf_counter() - start
        print(f"Integer sum warm-up (includes compile): {warmup_time:.3f} seconds")
        results["integer_sum_10M_warmup"] = warmup_time
        
        # Compiled run
        start = time.perf_counter()
        result = numba_sum(10_000_000)
        elapsed = time.perf_counter() - start
        print(f"Integer sum compiled (10M): {elapsed:.3f} seconds")
        results["integer_sum_10M_compiled"] = elapsed)
        
        @numba.jit(nopython=True, parallel=True)
        def numba_parallel_sum(arr):
            total = 0.0
            for i in numba.prange(len(arr)):
                total += arr[i] ** 2
            return total
        
        arr = np.random.randn(50_000_000)
        
        # Warm-up
        start = time.perf_counter()
        result = numba_parallel_sum(arr)
        warmup_time = time.perf_counter() - start
        print(f"Parallel sum warm-up (50M): {warmup_time:.3f} seconds")
        results["parallel_sum_50M_warmup"] = warmup_time
        
        # Compiled
        start = time.perf_counter()
        result = numba_parallel_sum(arr)
        elapsed = time.perf_counter() - start
        print(f"Parallel sum compiled (50M): {elapsed:.3f} seconds")
        results["parallel_sum_50M_compiled"] = elapsed
        
        RESULTS["benchmarks"]["numba"] = results
        print()
        
    except ImportError as e:
        print(f"Numba not available: {e}")
        RESULTS["benchmarks"]["numba"] = {"error": str(e)}
    except Exception as e:
        print(f"Numba benchmark failed: {e}")
        RESULTS["benchmarks"]["numba"] = {"error": str(e)})

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HARDWARE BENCHMARK FOR CI RUNNER")
    print("=" * 60 + "\n")
    
    get_cpu_info()
    benchmark_cpu_pure_python()
    benchmark_cpu_numpy()
    benchmark_numba()
    benchmark_gpu_jax()
    
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    # Save results to JSON
    output_file = "benchmark_results_baremetal.json"
    with open(output_file, 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nResults saved to {output_file}")
