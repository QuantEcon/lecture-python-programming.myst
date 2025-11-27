"""
Profile lax.scan performance on GPU vs CPU to investigate synchronization overhead.

This script helps diagnose why lax.scan with many lightweight iterations
performs poorly on GPU (81s) compared to CPU (0.06s).

Usage:
    # Basic timing comparison
    python profile_lax_scan.py

    # With NVIDIA Nsight Systems (requires nsys installed)
    nsys profile -o lax_scan_profile --trace=cuda,nvtx python profile_lax_scan.py --nsys

    # With JAX profiler (view with TensorBoard)
    python profile_lax_scan.py --jax-profile

    # With XLA debug dumps
    python profile_lax_scan.py --xla-dump

Requirements:
    - JAX with CUDA support: pip install jax[cuda12]
    - For Nsight: NVIDIA Nsight Systems (https://developer.nvidia.com/nsight-systems)
    - For TensorBoard: pip install tensorboard tensorboard-plugin-profile
"""

import argparse
import os
import time
from functools import partial

def setup_xla_dump(dump_dir="/tmp/xla_dump"):
    """Enable XLA debug dumps before importing JAX."""
    os.makedirs(dump_dir, exist_ok=True)
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
    print(f"XLA dumps will be written to: {dump_dir}")

def main():
    parser = argparse.ArgumentParser(description="Profile lax.scan GPU performance")
    parser.add_argument("--nsys", action="store_true", 
                        help="Run in Nsight Systems compatible mode (smaller n)")
    parser.add_argument("--jax-profile", action="store_true",
                        help="Enable JAX profiler (view with TensorBoard)")
    parser.add_argument("--xla-dump", action="store_true",
                        help="Dump XLA HLO for analysis")
    parser.add_argument("-n", "--iterations", type=int, default=10_000_000,
                        dest="n", help="Number of iterations (default: 10M)")
    parser.add_argument("--profile-dir", type=str, default="/tmp/jax-trace",
                        help="Directory for JAX profile output")
    args = parser.parse_args()

    # Setup XLA dump before importing JAX
    if args.xla_dump:
        setup_xla_dump()

    # Now import JAX
    import jax
    import jax.numpy as jnp
    from jax import lax

    print("=" * 60)
    print("lax.scan GPU Performance Profiling")
    print("=" * 60)
    
    # Show device info
    print(f"\nJAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.devices()[0]}")
    
    # Reduce n for Nsight profiling to keep trace manageable
    n = 100_000 if args.nsys else args.n
    print(f"\nIterations (n): {n:,}")

    # Define the functions
    @partial(jax.jit, static_argnums=(1,))
    def qm_jax_default(x0, n, α=4.0):
        """lax.scan on default device (GPU if available)."""
        def update(x, t):
            x_new = α * x * (1 - x)
            return x_new, x_new
        _, x = lax.scan(update, x0, jnp.arange(n))
        return jnp.concatenate([jnp.array([x0]), x])

    cpu = jax.devices("cpu")[0]
    
    @partial(jax.jit, static_argnums=(1,), device=cpu)
    def qm_jax_cpu(x0, n, α=4.0):
        """lax.scan forced to CPU."""
        def update(x, t):
            x_new = α * x * (1 - x)
            return x_new, x_new
        _, x = lax.scan(update, x0, jnp.arange(n))
        return jnp.concatenate([jnp.array([x0]), x])

    # Warm up (compilation)
    print("\n--- Compilation (warm-up) ---")
    print("Compiling default device version...", end=" ", flush=True)
    t0 = time.perf_counter()
    _ = qm_jax_default(0.1, n).block_until_ready()
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    print("Compiling CPU version...", end=" ", flush=True)
    t0 = time.perf_counter()
    _ = qm_jax_cpu(0.1, n).block_until_ready()
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    # Profile with JAX profiler if requested
    if args.jax_profile:
        print(f"\n--- JAX Profiler (output: {args.profile_dir}) ---")
        os.makedirs(args.profile_dir, exist_ok=True)
        
        jax.profiler.start_trace(args.profile_dir)
        
        # Run both versions while profiling
        print("Profiling default device version...")
        result_default = qm_jax_default(0.1, n).block_until_ready()
        
        print("Profiling CPU version...")
        result_cpu = qm_jax_cpu(0.1, n).block_until_ready()
        
        jax.profiler.stop_trace()
        print(f"\nProfile saved. View with:")
        print(f"  tensorboard --logdir={args.profile_dir}")

    # Timing runs
    print("\n--- Timing Runs (post-compilation) ---")
    
    # Default device (GPU if available)
    print(f"\nDefault device ({jax.devices()[0]}):")
    times_default = []
    for i in range(3):
        t0 = time.perf_counter()
        result = qm_jax_default(0.1, n).block_until_ready()
        elapsed = time.perf_counter() - t0
        times_default.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.6f}s")
    
    # CPU
    print(f"\nCPU (forced with device=cpu):")
    times_cpu = []
    for i in range(3):
        t0 = time.perf_counter()
        result = qm_jax_cpu(0.1, n).block_until_ready()
        elapsed = time.perf_counter() - t0
        times_cpu.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.6f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_default = sum(times_default) / len(times_default)
    avg_cpu = sum(times_cpu) / len(times_cpu)
    print(f"Default device avg: {avg_default:.6f}s")
    print(f"CPU avg:            {avg_cpu:.6f}s")
    print(f"Ratio (default/cpu): {avg_default/avg_cpu:.1f}x")
    
    if avg_default > avg_cpu * 10:
        print("\n⚠️  GPU is significantly slower than CPU!")
        print("   This confirms the lax.scan synchronization overhead issue.")
    elif avg_default < avg_cpu:
        print("\n✓ GPU is faster (unexpected for this workload)")
    else:
        print("\n~ Performance is similar")

    if args.xla_dump:
        print(f"\nXLA dumps written to /tmp/xla_dump/")
        print("Look for .txt files with HLO representation")

    if args.nsys:
        print("\nNsight Systems trace will be saved as lax_scan_profile.nsys-rep")
        print("View with: nsys-ui lax_scan_profile.nsys-rep")

if __name__ == "__main__":
    main()
