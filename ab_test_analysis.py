#!/usr/bin/env python3
import subprocess
import re
import os

os.chdir('/home/shadeform/Documents/modular')
os.environ['KERNEL_BENCHMARKS_ROOT'] = '/home/shadeform/Documents/modular/max/kernels/benchmarks'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

sizes = [262144, 524288, 1048576, 2097152, 4194304]
results_baseline = {}
results_cpasync = {}

def extract_gbs(output):
    """Extract GB/s from benchmark output"""
    # Look for the data row (contains "allreduce" in name, has 4 columns)
    lines = output.split('\n')
    for i, line in enumerate(lines):
        # Data row format: | allreduce-... | 0.069 | 100 | 3.792 |
        if 'allreduce' in line.lower() and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            # Format: | name | met (ms) | iters | DataMovement (GB/s) |
            # So parts[0] is empty, parts[1] is name, parts[2] is ms, parts[3] is iters, parts[4] is GB/s
            if len(parts) >= 5:
                gbs_str = parts[4]
                try:
                    gbs = float(gbs_str)
                    # Make sure it's a reasonable GB/s value (between 0.1 and 1000)
                    if 0.1 <= gbs <= 1000:
                        return gbs
                except ValueError:
                    pass
    return None

print("=== A/B TEST: Small Payloads (1-stage kernel) ===\n")

# Baseline
print("BASELINE (cp.async DISABLED):")
os.environ['MODULAR_MAX_ALLREDUCE_CPASYNC'] = '0'
for size in sizes:
    cmd = ['./bazelw', 'run', '//max/kernels/benchmarks:gpu/bench_allreduce', '--', 
           f'--num_bytes={size}', '--NUM_GPUS=2', '--dtype=fp32']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    gbs = extract_gbs(result.stdout)
    if gbs is not None:
        results_baseline[size] = gbs
        print(f"  {size:8d} bytes: {gbs:.3f} GB/s")
    else:
        print(f"  {size:8d} bytes: FAILED")

# cp.async
print("\nWITH cp.async (ENABLED):")
os.environ['MODULAR_MAX_ALLREDUCE_CPASYNC'] = '1'
for size in sizes:
    cmd = ['./bazelw', 'run', '//max/kernels/benchmarks:gpu/bench_allreduce', '--', 
           f'--num_bytes={size}', '--NUM_GPUS=2', '--dtype=fp32']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    gbs = extract_gbs(result.stdout)
    if gbs is not None:
        results_cpasync[size] = gbs
        print(f"  {size:8d} bytes: {gbs:.3f} GB/s")
    else:
        print(f"  {size:8d} bytes: FAILED")

# Analysis
print("\n=== IMPROVEMENT ANALYSIS ===")
print(f"{'Size':>10} | {'Baseline (GB/s)':>15} | {'cp.async (GB/s)':>15} | {'Speedup':>8} | {'% Gain':>8}")
print("-" * 75)

speedups = []
for size in sizes:
    if size in results_baseline and size in results_cpasync:
        b = results_baseline[size]
        c = results_cpasync[size]
        speedup = c / b
        pct = (c - b) / b * 100
        speedups.append(speedup)
        print(f"{size:10d} | {b:15.3f} | {c:15.3f} | {speedup:8.2f}x | {pct:7.1f}%")

if speedups:
    avg_speedup = sum(speedups) / len(speedups)
    avg_pct = (avg_speedup - 1) * 100
    print("-" * 75)
    print(f"{'AVERAGE':>10} | {'':15} | {'':15} | {avg_speedup:8.2f}x | {avg_pct:7.1f}%")
    
    if avg_speedup > 1.0:
        print(f"\n✓ cp.async provides {avg_pct:.1f}% improvement on average")
    elif avg_speedup < 1.0:
        print(f"\n✗ cp.async shows {abs(avg_pct):.1f}% regression on average")
    else:
        print(f"\n= cp.async shows no significant difference")

