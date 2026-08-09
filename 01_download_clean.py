"""
01_download_clean.py
============================================================
數據獲取與預處理（支援 adult / heart 資料集）

  1. 下載資料集（有快取則略過）
  2. 清洗：欄位命名、去空白/尾端句點、缺失值標準化、去重複
  3. 特徵工程：數值特徵 Min-Max 縮放、類別特徵 Label Encoding、標籤二值化
  4. 儲存 data/{name}_arrays.npz（供 02–05 腳本使用）

執行：python3 01_download_clean.py [--datasets adult,heart]
輸出：data/adult_arrays.npz、data/heart_arrays.npz
============================================================
"""
import argparse
import json
import os
from datasets import DATASETS, DATA_DIR, prepare_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="下載並前處理表格資料集")
    ap.add_argument("--datasets", default="adult,heart",
                    help="逗號分隔的資料集名稱（預設 adult,heart）")
    args = ap.parse_args()

    summaries = {}
    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        if name not in DATASETS:
            print(f"[警告] 忽略未知資料集: {name}")
            continue
        summaries[name] = prepare_dataset(name, seed=42)

    # 寫入摘要，方便其他腳本/文件引用
    with open(os.path.join(DATA_DIR, "dataset_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("\n已儲存 data/dataset_summary.json")


if __name__ == "__main__":
    main()
