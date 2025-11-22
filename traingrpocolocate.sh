#!/usr/bin/env bash
set -euo pipefail

################################
# 1. 统一环境变量
################################
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=12
export VLLM_USE_V1=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30600
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export PYTHONPATH="/root/open-r1/src:/root/TRL-ADPO${PYTHONPATH:+:$PYTHONPATH}"

################################
# 2. 直接启动 GRPO 训练（colocate 模式）
################################
echo "🚀 开始 GRPO Baseline 训练（vLLM colocate 模式）..."

# 使用 4 张卡的配置文件
accelerate launch --config_file /root/open-r1/recipes/accelerate_configs/zero2.yaml \
    --num_processes 4 \
    /root/open-r1/src/open_r1/grpo.py \
    --config /root/open-r1/recipes/Qwen3/grpo/config_qwen3-1_6b_baseline.yaml

echo "✅ GRPO 训练完成！"
