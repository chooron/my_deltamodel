#!/usr/bin/env bash
# 并行率定 common 水文模型（首字母小写，使用 config_dcommon_calibrate.yaml）
# 用法: bash run_calibrations_common_parallel.sh

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
# 组1: 小参数 (1-4 params)
GROUP1=(
    "collie1"    # 1
    "wetland"    # 2
    "alpine1"    # 4
    "penman"     # 4
    "collie2"    # 4
)
# 组2: 中小参数 (5-7 params)
GROUP2=(
    "mopex1"     # 5
    "hymod"      # 5
    "us1"        # 5
    "collie3"    # 6
    "tcm"        # 6
    "newzealand1" # 6
    "susannah1"  # 6
    "susannah2"  # 6
    "alpine2"    # 6
    "mopex2"     # 7
    "simhyd"     # 7
    "topmodel"   # 7
)
# 组3: 中大参数 (8-10 params)
GROUP3=(
    "australia"  # 8
    "gsfb"       # 8
    "mopex3"     # 8
    "vic"        # 10
)
# 组4: 大参数 (12-15 params)
GROUP4=(
    "tank"       # 12
    "xinanjiang" # 12
    "hbv96"      # 15
    "modhydrolog" # 15
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

    "$PYTHON" -u "$SCRIPT_DIR/calibrate_common_parallel.py" \
        --model "$model" 2>&1 | tee "$SCRIPT_DIR/logs/${model}_common.log"

    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 完成: $model"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 失败: $model (查看 logs/${model}_common.log)"
    fi
}

echo "========================================"
echo "并行率定 common 模型 (GPU 0)"
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
echo "所有 common 模型任务完成!"
echo "========================================"
