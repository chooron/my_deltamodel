#!/bin/bash

# 遍历不同的alpha值执行run_model.py（并行执行）
# Alpha values: 0, 0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.5, 1
# cd /workspace/my_deltamodel/project/flex_mopex
# ./run_alpha_experiments.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# alphas=(0 0.1 0.01 0.03 0.05 0.07 0.001 0.003 0.005 0.007)
alphas=(0)

pids=()

for alpha in "${alphas[@]}"; do
    echo "Starting experiment with alpha = $alpha"
    python run_model.py "$alpha" &
    pids+=($!)
done

echo "All experiments launched, waiting for completion..."

failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run with alpha = ${alphas[$i]}"
        failed=1
    else
        echo "Completed alpha = ${alphas[$i]}"
    fi
done

if [ $failed -ne 0 ]; then
    echo "Some experiments failed!"
    exit 1
fi

echo "All experiments completed successfully!"
