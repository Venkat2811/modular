# cp.async A/B Test Results

## Test Configuration
- **Hardware**: 2x NVIDIA H100 GPUs (HBM3, NVLink)
- **Test Sizes**: 256KB, 512KB, 1MB, 2MB, 4MB (1-stage kernel range)
- **Data Type**: fp32
- **Baseline**: Regular 1-stage kernel (cp.async disabled)
- **Test**: cp.async double-buffering kernel (cp.async enabled)

## Results

| Size | Baseline (GB/s) | cp.async (GB/s) | Speedup | % Gain |
|------|----------------|-----------------|--------|--------|
| 256KB | 3.783 | 3.781 | 1.00x | -0.1% |
| 512KB | 3.516 | 3.515 | 1.00x | -0.0% |
| 1MB | 3.590 | 3.593 | 1.00x | +0.1% |
| 2MB | 3.632 | 3.631 | 1.00x | -0.0% |
| 4MB | 3.650 | 3.649 | 1.00x | -0.0% |
| **AVERAGE** | **3.634** | **3.634** | **1.00x** | **-0.0%** |

## Analysis

### Key Findings
1. **No Performance Improvement**: cp.async kernel shows essentially identical performance to baseline (0.0% average difference)
2. **Consistent Across Sizes**: No improvement observed across the tested size range (256KB - 4MB)
3. **Slight Variance**: Differences are within measurement noise (±0.1%)

### Possible Explanations

1. **Prefetch Not Overlapping**: The prefetch may not be effectively overlapping with computation
   - Small chunk sizes may not provide enough work to hide memory latency
   - Synchronization overhead (`async_copy_wait_group`, `barrier()`) may be negating benefits

2. **Memory Bandwidth Saturation**: At ~3.6 GB/s, we're far below NVLink bandwidth (~900 GB/s theoretical)
   - This suggests the bottleneck is not memory bandwidth but something else (kernel launch overhead, synchronization, etc.)

3. **Kernel Launch Overhead**: For small sizes, kernel launch overhead may dominate
   - The cp.async kernel has more complex synchronization which may add overhead

4. **Workload Characteristics**: Small all-reduce operations may not benefit from prefetching
   - The computation (simple addition) is very fast
   - Memory access pattern may already be optimal for regular loads

### Next Steps

1. **Profile with CUDA Profiler**: Use `nsys profile` to verify:
   - Which kernel is actually executing
   - Memory bandwidth utilization
   - Whether prefetch overlaps with computation
   - Synchronization overhead

2. **Test Larger Sizes**: Try sizes that use 2-stage kernel (>= 8MB) to see if prefetch helps there

3. **Simplify Synchronization**: Reduce `barrier()` calls and optimize `async_copy_wait_group` placement

4. **Consider Alternative Approaches**:
   - TMA (Tensor Memory Accelerator) for H100
   - `multimem_ld_reduce` hardware instruction
   - PDL (Programmatic Dependent Launch) overlap

## Conclusion

The cp.async double-buffering implementation does not provide measurable performance improvement for small payloads (256KB - 4MB) on 2x H100 GPUs. The overhead of additional synchronization appears to negate any benefits from prefetching, or the prefetch is not effectively overlapping with computation.

Further investigation with profiling tools is needed to understand the root cause and determine if cp.async can be optimized or if alternative approaches should be pursued.

## Nsight Systems Profiling (256KB case)

- Baseline capture: `profiles/nsys_allreduce_256k_baseline.nsys-rep`
- cp.async capture: `profiles/nsys_allreduce_256k_cpasync.nsys-rep`
- Both runs launch two kernels per iteration (`comm_allreduce__allreduce_1stage...`) with ~69µs average duration and identical instance counts (220), confirming the cp.async path follows the same launch cadence as the baseline.
- GPU memory activity is minimal (only ~0.52 MB memcpy per direction and ~1.7 MB memset total), indicating the benchmark is compute/launch bound rather than bandwidth bound for these payloads.
- CUDA API statistics show the cp.async run roughly doubles the average `cuLaunchKernelEx` overhead (6.8µs vs. 3.2µs) while reducing `cuEventSynchronize` time slightly, reinforcing that additional synchronization/control logic offsets any prefetch benefit.
- Recommendation: inspect kernel timelines in Nsight GUI to verify whether asynchronous copy stages overlap with computation, and experiment with larger chunk sizes or reduced synchronization to seek measurable gains.

## Nsight Systems Profiling (16MB case)

- Baseline capture: `profiles/nsys_allreduce_16m_baseline.nsys-rep`
- cp.async capture: `profiles/nsys_allreduce_16m_cpasync.nsys-rep`
- The 2-stage kernels dominate runtime in both runs (`comm_allreduce__allreduce_2stage...`, 220 instances at ~4.78 ms each) with no detectable duration delta between baseline and cp.async.
- Host-device and device-host memcpy volumes rise to ~16.8 MB per transfer (as expected), yet measured bandwidth remains ~3.51 GB/s and identical across both runs.
- CUDA API stats are again dominated by long `cuEventSynchronize` calls, while cp.async introduces slightly higher `cuLaunchKernelEx` overhead without improving kernel duration.
- Conclusion: even in the larger, bandwidth-oriented regime the cp.async path fails to overlap data movement with computation; future experiments should target pipeline restructuring (e.g., larger per-thread tiles, multi-stage prefetch, or TMA/multimem alternatives).

## Trace Inspection (GPU timeline)

- Exported GPU traces: `profiles/nsys_allreduce_256k_{baseline,cpasync}_gpu_trace_cuda_gpu_trace.csv` and `profiles/nsys_allreduce_16m_{baseline,cpasync}_gpu_trace_cuda_gpu_trace.csv`.
- For each GPU and stream, kernel start times are always greater than or equal to the end time of the previous kernel in the trace (verified via script), so there is **no intra-GPU overlap** between successive `_allreduce_*` launches in either configuration.
- Memory copies appear before the first kernel launch and finish before computation begins; there are no interleaved memcpy/compute segments that would suggest asynchronous staging at the GPU timeline level.
- Because cp.async executes within the kernel, Nsight Systems cannot show instruction-level overlap directly, but the absence of any runtime reduction (and matching kernel durations) indicates the prefetch does not translate into observable concurrency.

