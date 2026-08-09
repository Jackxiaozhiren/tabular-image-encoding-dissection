"""
02_visualize_encoding.py
============================================================
特徵編碼與可視化（支援 adult / heart 資料集）

  使用 tabular_to_image.py 的三種編碼，各取數個樣本繪製編碼圖像：
    M1  特徵列熱圖（灰度）
    M2  相關性排序 2D 拼貼（IGTD 風格）
    M3  多通道編碼（本研究提出，RGB 三通道拆解）
  另繪製「編碼方法示意圖」，把 M1 / M2 / M3 的結構畫在同一張圖上。

執行：python3 02_visualize_encoding.py [--dataset adult]
輸出：figures/fig_encoding_M1_{name}.png、fig_encoding_M2_{name}.png、
      fig_encoding_M3_{name}.png、fig_encoding_overview_{name}.png
============================================================
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文字型設定（macOS 內建；Linux 可改用 Noto Sans CJK TC）
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "PingFang SC",
                                   "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from datasets import DATASETS, DATA_DIR
from tabular_to_image import encode_m1, encode_m2, encode_m3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

WIDTH, HEIGHT = 64, 16
SEED = 42


def main():
    ap = argparse.ArgumentParser(description="產生編碼圖像範例")
    ap.add_argument("--dataset", default="adult", choices=list(DATASETS.keys()))
    args = ap.parse_args()
    name = args.dataset
    cfg = DATASETS[name]

    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    X_train_num, y_train = npz["X_train_num"], npz["y_train"]
    X_train_cat = npz["X_train_cat"]
    numeric, categorical = cfg["numeric"], cfg["categorical"]

    # 固定種子，各取 3 正例 + 3 負例樣本（可重現）
    rng = np.random.default_rng(SEED)
    pos_idx, neg_idx = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
    sample_idx = np.concatenate([
        rng.choice(neg_idx, min(3, len(neg_idx)), replace=False),
        rng.choice(pos_idx, min(3, len(pos_idx)), replace=False),
    ])
    y_sample = y_train[sample_idx]

    # ---------- M1：特徵列熱圖（灰度） ----------
    imgs = [encode_m1(X_train_num[i], width=WIDTH)[0] for i in sample_idx]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax, img, i, yy in zip(axes.ravel(), imgs, sample_idx, y_sample):
        ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"樣本 #{i}  標籤={yy}")
        ax.set_yticks(range(len(numeric)))
        ax.set_yticklabels(numeric, fontsize=8)
        ax.set_xticks([])
    plt.tight_layout()
    fig.suptitle(f"M1 特徵列熱圖（灰度）— {cfg['display']}", y=1.01)
    plt.savefig(os.path.join(FIG_DIR, f"fig_encoding_M1_{name}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"[1/4] 已儲存 fig_encoding_M1_{name}.png")

    # ---------- M2：相關性排序 2D 拼貼 ----------
    corr = np.array([np.corrcoef(X_train_num[:, k], y_train)[0, 1]
                     for k in range(X_train_num.shape[1])])
    imgs = [encode_m2(X_train_num[i], corr, grid=4)[0] for i in sample_idx]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, img, i, yy in zip(axes.ravel(), imgs, sample_idx, y_sample):
        ax.imshow(img, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"樣本 #{i}  標籤={yy}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.suptitle(f"M2 相關性排序 2D 拼貼（4x4）— {cfg['display']}", y=1.01)
    plt.savefig(os.path.join(FIG_DIR, f"fig_encoding_M2_{name}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"[2/4] 已儲存 fig_encoding_M2_{name}.png")

    # ---------- M3：多通道編碼（本研究提出） ----------
    def one_hot_vec(i, c):
        n_cat = int(X_train_cat[:, c].max()) + 1
        oh = np.zeros(n_cat)
        oh[int(X_train_cat[i, c])] = 1.0
        return oh

    # 權重重排：以 XGBoost 對數值特徵的重要度（若已計算）或均等權重
    imp_path = os.path.join(DATA_DIR, f"{name}_xgb_importance.npz")
    if os.path.exists(imp_path):
        weight_map = np.load(imp_path)["importance_numeric"]
    else:
        weight_map = np.ones(len(numeric)) / len(numeric)

    imgs = [encode_m3(X_train_num[i], [one_hot_vec(i, c)
                                       for c in range(len(categorical))],
                      width=WIDTH, height=HEIGHT, weight_map=weight_map)
            for i in sample_idx[:3]]
    channel_names = ["R：數值特徵熱圖", "G：類別 one-hot 區塊", "B：權重重排熱圖"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for r, (img, i) in enumerate(zip(imgs, sample_idx[:3])):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(img[c], cmap="gray" if c != 1 else "magma",
                      aspect="auto", vmin=0, vmax=1)
            ax.set_title(channel_names[c] if r == 0 else f"樣本 #{i}")
            ax.set_xticks([])
            if c == 0:
                ax.set_ylabel(f"樣本 #{i}\n標籤={y_train[i]}")
            if r == 2:
                ax.set_yticks([])
    for ax in axes[0]:
        ax.set_yticks(range(HEIGHT))
        ax.set_yticklabels([f"f{r}" for r in range(HEIGHT)], fontsize=6)
    plt.tight_layout()
    fig.suptitle(f"M3 多通道編碼 — R 數值 / G 類別 one-hot / B 權重重排（{cfg['display']}）",
                 y=1.01)
    plt.savefig(os.path.join(FIG_DIR, f"fig_encoding_M3_{name}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"[3/4] 已儲存 fig_encoding_M3_{name}.png")

    # ---------- 編碼方法示意圖 ----------
    img_m1 = encode_m1(X_train_num[sample_idx[0]], width=WIDTH)[0]
    img_m2 = encode_m2(X_train_num[sample_idx[0]], corr, grid=4)[0]
    img_m3 = encode_m3(X_train_num[sample_idx[0]],
                       [one_hot_vec(sample_idx[0], c)
                        for c in range(len(categorical))],
                       width=WIDTH, height=HEIGHT, weight_map=weight_map)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img_m1, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("(a) M1 特徵列熱圖")
    axes[0].set_ylabel("特徵"); axes[0].set_xlabel("像素寬度 W")
    axes[0].set_yticks([])
    axes[1].imshow(img_m2, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    axes[1].set_title("(b) M2 相關性排序拼貼")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    rgb = np.stack([img_m3[c] for c in range(3)], axis=-1)
    axes[2].imshow(rgb, aspect="auto")
    axes[2].set_title("(c) M3 多通道編碼 (RGB)")
    axes[2].set_ylabel("通道區塊"); axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_encoding_overview_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[4/4] 已儲存 fig_encoding_overview_{name}.png")


if __name__ == "__main__":
    main()
