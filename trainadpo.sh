#!/usr/bin/env bash
set -euo pipefail

################################
# 1. 统一环境变量
################################
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=12
export VLLM_USE_V1=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30500
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export PYTHONPATH="/root/open-r1/src:/root/TRL-ADPO${PYTHONPATH:+:$PYTHONPATH}"

################################
# 2. 启动 vLLM serving（后台）
################################
LOG_DIR=/root/logs
mkdir -p "$LOG_DIR"
VLLM_LOG=$LOG_DIR/vllm_adpo_$(date +%Y%m%d_%H%M%S).log

# 使用独立的 vLLM server，端口 8000，使用 GPU 1,2
CUDA_VISIBLE_DEVICES=1,2 \
  trl vllm-serve \
  --model Qwen/Qwen3-1.7B \
  --gpu-memory-utilization 1.0 \
  --tensor-parallel-size 2 \
  --port 8000 2>&1 | tee "$VLLM_LOG" &
VLLM_PID=$!

################################
# 3. 等待端口就绪…
################################
echo "⏳ 正在等待 vLLM 服务就绪..."
TIMEOUT=1000
for ((i=1;i<=TIMEOUT;i++)); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    echo "✅ vLLM /health OK，继续后续任务。"
    break
  fi
  if (( i % 10 == 0 )); then
    echo "   ...已等待 ${i} 秒"
  fi
  sleep 1
done

if (( i > TIMEOUT )); then
  echo "❌ vLLM 启动超时，请检查日志: $VLLM_LOG"
  kill $VLLM_PID || true
  exit 1
fi

################################
# 4. 启动 ADPO 训练
################################
echo "🚀 开始 ADPO Server 模式训练..."

# 使用新的 server 模式配置，训练使用 GPU 3,4
CUDA_VISIBLE_DEVICES=3,4 \
  accelerate launch --config_file /root/open-r1/recipes/accelerate_configs/zero2.yaml \
    /root/open-r1/src/open_r1/adpo.py \
    --config /root/open-r1/recipes/Qwen3/adpo/config_qwen3-1_6b_server.yaml \
    --vllm_server_base_url http://127.0.0.1:8000

################################
# 5. 训练结束，清理 vLLM
################################
echo "🏁 训练结束，正在关闭 vLLM (PID=$VLLM_PID)..."
kill $VLLM_PID || true
wait $VLLM_PID || true
echo "✅ vLLM 已关闭。"
