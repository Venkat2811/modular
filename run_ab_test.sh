#!/bin/bash
cd /home/shadeform/Documents/modular
export KERNEL_BENCHMARKS_ROOT=$(pwd)/max/kernels/benchmarks
export CUDA_VISIBLE_DEVICES=0,1

sizes=(262144 524288 1048576 2097152 4194304)

echo "=== A/B TEST: Small Payloads (1-stage kernel) ==="
echo ""

echo "BASELINE (cp.async DISABLED):"
export MODULAR_MAX_ALLREDUCE_CPASYNC=0
for size in "${sizes[@]}"; do
    output=$(./bazelw run //max/kernels/benchmarks:gpu/bench_allreduce -- --num_bytes=$size --NUM_GPUS=2 --dtype=fp32 2>&1)
    gbs=$(echo "$output" | grep "DataMovement" | tail -1 | awk -F'|' '{print $4}' | xargs)
    echo "  $size bytes: $gbs GB/s"
done

echo ""
echo "WITH cp.async (ENABLED):"
export MODULAR_MAX_ALLREDUCE_CPASYNC=1
for size in "${sizes[@]}"; do
    output=$(./bazelw run //max/kernels/benchmarks:gpu/bench_allreduce -- --num_bytes=$size --NUM_GPUS=2 --dtype=fp32 2>&1)
    gbs=$(echo "$output" | grep "DataMovement" | tail -1 | awk -F'|' '{print $4}' | xargs)
    echo "  $size bytes: $gbs GB/s"
done

