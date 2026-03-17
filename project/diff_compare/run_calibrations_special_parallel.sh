#!/usr/bin/env bash
# 并行率定 special 水文模型（首字母大写，使用 config_dspecial_calibrate.yaml）
# 用法: bash run_calibrations_special_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

UV_VENV="/workspace/my_deltamodel/.venv"
if [ -f "$UV_VENV/bin/activate" ]; then
    source "$UV_VENV/bin/activate"
fi

: "${PYTHON:=python3}"
MAX_PARALLEL=1

# 按参数量分4组，避免同时跑大参数模型导致OOM
# 每次从各组各取1个模型并行，保证4个槽内存压力均衡
# 组1: 小参数 (4-6 params)
GROUP1=(
    "Gr4j"       # 4
    "Mopex5"     # 5
    "Ihacres"    # 6
)
# 组2: 中参数 (7-8 params)
GROUP2=(
    "Hillslope"  # 7
    "Smar"       # 8
    "Plateau"    # 8
    "Newzealand2" # 8
)
# 组3: 中大参数 (9-10 params)
GROUP3=(
    "Flexb"      # 9
    "Flexi"      # 10
    "Mopex4"     # 10
)
# 组4: 大参数 (12 params)
GROUP4=(
    "Flexis"     # 12
)

# 交错合并：每轮从各组各取1个，保证并行时参数量均衡
MODELS=()
max_len=$(( ${#GROUP1[@]} > ${#GROUP2[@]} ? ${#GROUP1[@]} : ${#GROUP2[@]} ))
max_len=$(( max_len > ${#GROUP3[@]} ? max_len : ${#GROUP3[@]} ))
max_len=$(( max_len > ${#GROUP4[@]} ? max_len : ${#GROUP4[@]} ))
for (( i=0; i<max_len; i++ )); do
    [ $i -lt ${#GROUP1[@]} ] && MODELS+=("${GROUP1[$i]}")
    [ $i -lt ${#GROUP2[@]} ] && MODELS+=("${GROUP2[$i]}")
    [ $i -lt ${#GROUP3[@]} ] && MODELS+=("${GROUP3[$i]}")
    [ $i -lt ${#GROUP4[@]} ] && MODELS+=("${GROUP4[$i]}")
done

run_model() {
    local model=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始率定: $model"
    mkdir -p "$SCRIPT_DIR/logs"

    "$PYTHON" "$SCRIPT_DIR/calibrate_special_parallel.py" \
        --model "$model" \
        > "$SCRIPT_DIR/logs/${model}_special.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 完成: $model"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 失败: $model (查看 logs/${model}_special.log)"
    fi
}

echo "========================================"
echo "并行率定 special 模型 (GPU 1)"
echo "最大并行数: $MAX_PARALLEL"
echo "模型总数: ${#MODELS[@]}"
echo "========================================"

mkdir -p logs
export -f run_model
export PYTHON
export SCRIPT_DIR

if command -v parallel &> /dev/null; then
    printf "%s\n" "${MODELS[@]}" | parallel -j "$MAX_PARALLEL" run_model {}
else
    printf "%s\n" "${MODELS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_model "$@"' _ {}
fi

echo ""
echo "========================================"
echo "所有 special 模型任务完成!"
echo "========================================"
