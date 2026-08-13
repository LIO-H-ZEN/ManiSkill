#!/bin/bash
# 自动监控 Piper PickAnything 训练进度

RUN_DIR="/home/liu/Desktop/mani_skill_repo/runs/piper-PickAnything-v1-state-7-walltime_efficient"
LOG_PATTERN="/home/liu/Desktop/logs/piper_pickanything_state_train_7_*.log"
PID=376997

echo "=========================================="
echo " Piper PickAnything 训练监控"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================

"

# 1. 进程状态
echo "--- 进程状态 ---"
if ps -p $PID > /dev/null 2>&1; then
    ps -p $PID -o pid,stat,%cpu,%mem,etime --no-headers
else
    echo "❌ 进程已停止！"
fi

# 2. GPU
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null

# 3. 最新训练日志
echo ""
echo "--- 最新日志 ---"
tail -10 $(ls -t $LOG_PATTERN 2>/dev/null | head -1)

# 4. eval 成功率汇总
echo ""
echo "--- Eval 成功率历史 ---"
grep "eval_success_once_mean" $(ls -t $LOG_PATTERN 2>/dev/null | head -1) | tail -10

# 5. eval_return 趋势
echo ""
echo "--- Eval Return 历史 ---"
grep "eval_return_mean" $(ls -t $LOG_PATTERN 2>/dev/null | head -1) | tail -10

# 6. 最新 eval 视频
echo ""
echo "--- 最新视频 ---"
ls -lt "$RUN_DIR/videos/" 2>/dev/null | head -3

# 7. 进度估算
echo ""
echo "--- 进度 ---"
CURRENT_STEP=$(grep "global_step=" $(ls -t $LOG_PATTERN 2>/dev/null | head -1) | tail -1 | grep -oP 'global_step=\K[0-9]+')
if [ -n "$CURRENT_STEP" ]; then
    PROGRESS=$(echo "scale=1; $CURRENT_STEP / 10000000 * 100" | bc)
    echo "当前步数: $CURRENT_STEP / 10,000,000 ($PROGRESS%)"
fi

echo ""
echo "=========================================="