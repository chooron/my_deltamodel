#!/bin/bash

# 遍历不同的alpha值执行run_model.py（并行执行，但交错启动）
# Alpha values: 0, 0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.5, 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

alphas=(0 0.1 0.01 0.03 0.05 0.07 0.001 0.003 0.005 0.007 0.5 1)
# alphas=(0)

pids=()
num_alphas=${#alphas[@]}

for i in "${!alphas[@]}"; do
    alpha="${alphas[$i]}"
    
    echo "Starting experiment with alpha = $alpha (Process $((i+1))/$num_alphas)"
    
    # 后台执行
    python run_model.py "$alpha" &
    pids+=($!)
    
    # 如果不是最后一个参数，则等待30秒再启动下一个
    if [ $i -lt $((num_alphas - 1)) ]; then
        echo "Waiting 30 seconds before launching the next one..."
        sleep 30
    fi
done

echo "------------------------------------------------"
echo "All experiments launched, waiting for all processes to complete..."
echo "------------------------------------------------"

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

# 脚本最后一行添加：
echo "All tasks done. Shutting down now..."
/usr/bin/shutdown