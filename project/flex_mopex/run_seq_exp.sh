#!/bin/bash

# 遍历不同的alpha值执行run_model.py（串行执行：一个跑完再跑下一个）
# Alpha values: 0, 0.1, 0.01, 0.03, 0.05, 0.07, 0.001, 0.003, 0.005, 0.007, 0.5, 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

alphas=(0 0.1 0.01 0.03 0.05 0.07 0.001 0.003 0.005 0.007 0.5 1)

num_alphas=${#alphas[@]}

for i in "${!alphas[@]}"; do
    alpha="${alphas[$i]}"
    
    echo "===================================================="
    echo "Starting experiment with alpha = $alpha (Task $((i+1))/$num_alphas)"
    echo "===================================================="
    
    # 串行执行：去掉 & 符号，脚本会等待 python 进程结束后再继续循环
    python run_model.py "$alpha"
    
    # 检查执行状态，如果出错则停止脚本（防止出错后仍然关机）
    if [ $? -ne 0 ]; then
        echo "Error: Experiment failed with alpha = $alpha. Aborting."
        exit 1
    fi
    
    echo "Successfully completed alpha = $alpha"
    echo "----------------------------------------------------"
done

echo "All experiments completed successfully!"