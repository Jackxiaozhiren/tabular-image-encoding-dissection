#!/bin/bash
# batch_train_new7.sh — 訓練 7 個新資料集的完整管線（背景批次）
# 依序：03 XGBoost+importances → 07 modern baselines → 04 CNN variants →
#       10 IGTD → 08 統計 → make_tables → 09 圖表
set -e
cd /Users/jackson/JDS论文
NEW="australian cmc ilpd segment vehicle spambase magic"
NUM_ONLY="segment vehicle spambase magic"

echo "========== 03 XGBoost baseline + importances =========="
for d in $NEW; do
  echo "---- 03 $d ----"
  python3 03_train_baseline.py --dataset $d || echo "[FAIL 03 $d]"
done

echo "========== 07 modern baselines (LightGBM/CatBoost/MLP/FT) =========="
python3 07_train_modern_baselines.py --datasets "$(echo $NEW | tr ' ' ',')" || echo "[FAIL 07]"

echo "========== 04 CNN variants =========="
FULL_TAGS="M1,M1c,M2,M3-full,M3-noG,M3-noB,M3-corrB,M3-shapB,M3-RG"
NUM_TAGS="M1,M2,M3-full,M3-noB,M3-corrB,M3-shapB,M3-RG"
for d in $NEW; do
  TAGS=$FULL_TAGS
  if echo " $NUM_ONLY " | grep -q " $d "; then TAGS=$NUM_TAGS; fi
  echo "---- 04 $d tags=$TAGS ----"
  python3 04_train_cnn.py --dataset $d --tags $TAGS || echo "[FAIL 04 $d]"
done

echo "========== 10 IGTD published encodings =========="
python3 10_published_encodings.py --datasets "$(echo $NEW | tr ' ' ',')" || echo "[FAIL 10]"

echo "========== 08 statistics (all 15 datasets) =========="
python3 08_extended_evaluate.py || echo "[FAIL 08]"

echo "========== make_tables =========="
python3 make_tables.py > /tmp/new_tables.txt 2>&1 || echo "[FAIL make_tables]"

echo "========== 09 figures =========="
python3 09_make_figures.py > /tmp/new_figures.log 2>&1 || echo "[FAIL 09]"

echo "========== BATCH DONE =========="
