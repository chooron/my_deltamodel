#!/bin/bash

# 遍历不同的alpha值执行run_model.py
# Alpha values: 0, 0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.5, 1
# cd /workspace/my_deltamodel/project/flex_mopex
# ./run_alpha_experiments.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# alphas=(0 0.1 0.01 0.03 0.05 0.07 0.001 0.003 0.005 0.007)
alphas=(0.006)

for alpha in "${alphas[@]}"; do
    echo "=========================================="
    echo "Running experiment with alpha = $alpha"
    echo "=========================================="

    python run_model.py "$alpha"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to run with alpha = $alpha"
        exit 1
    fi

    echo "Completed alpha = $alpha"
    echo ""
done

echo "All experiments completed successfully!"
