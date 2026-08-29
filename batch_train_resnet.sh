#!/bin/bash
# batch_train_resnet.sh — 第二骨架（ResNet）架構魯棒性實驗（可續跑）
# 訓練 13 個資訊量資料集 × 關鍵編碼 × 3 seeds；已存在的新鮮 npz 自動跳過。
set -u
cd "$(dirname "$0")"
LOG=/tmp/resnet_batch.log
MIXED="adult bank credit german telco sick heart australian cmc ilpd"
NUMERIC="wine spambase magic"
FAIL=""
TRAINED=0; SKIPPED=0

train() {  # ds tags [--balanced]
  local ds=$1 tags=$2; shift 2
  for tag in $tags; do
    local f="data/${ds}_cnn_${tag}_resnet_results.npz"
    if [ -f "$f" ]; then SKIPPED=$((SKIPPED+1)); continue; fi
    echo "---- ${ds} ${tag} $(date +%H:%M:%S) ----" | tee -a "$LOG"
    python3 04_train_cnn.py --dataset "$ds" --backbone resnet --tags "$tag" "$@" \
      >> "$LOG" 2>&1 \
      && { TRAINED=$((TRAINED+1)); echo "  OK ${ds} ${tag}" >> "$LOG"; } \
      || { echo "  FAIL ${ds} ${tag}" >> "$LOG"; FAIL="$FAIL $ds/$tag"; }
  done
}

echo "[$(date +%H:%M:%S)] 續跑 ResNet 批量（已存在者跳過）" | tee -a "$LOG"
for ds in $MIXED; do
  BAL=""; [ "$ds" = "heart" ] && BAL="--balanced"
  train "$ds" "M1 M1c M3-full M3-noG M3-RG" $BAL
done
for ds in $NUMERIC; do
  train "$ds" "M1 M3-full M3-RG"
done

echo "===== 完成 $(date +%H:%M:%S)：訓練 $TRAINED、跳過 $SKIPPED =====" | tee -a "$LOG"
echo "失敗: ${FAIL:-無}" | tee -a "$LOG"
echo "===== npz mtime 核對 =====" >> "$LOG"
for ds in $MIXED $NUMERIC; do
  tags="M1 M1c M3-full M3-noG M3-RG"; [ "$ds" = "wine" -o "$ds" = "spambase" -o "$ds" = "magic" ] && tags="M1 M3-full M3-RG"
  for tag in $tags; do
    f="data/${ds}_cnn_${tag}_resnet_results.npz"
    [ -f "$f" ] && echo "  $(stat -f '%Sm' "$f")  $f" >> "$LOG" || echo "  缺檔  $f" >> "$LOG"
  done
done
echo "LOG: $LOG"
