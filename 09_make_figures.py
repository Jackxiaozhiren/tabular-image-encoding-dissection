"""
09_make_figures.py
============================================================
阶段四：发表级矢量图生成（Neurocomputing 要求 EPS/PDF 矢量图）

  1. fig1_architecture.pdf   —— M3 编码示意（表格行 -> R/G/B 三通道 -> CNN -> 预测）
  2. fig_roc_{adult,heart,wine}.pdf  —— ROC 曲线（精选方法集）
  3. fig_pr_{adult,heart,wine}.pdf   —— PR 曲线
  4. fig_encodings.pdf       —— M1 / M1c / M3 编码示例对比（Adult）
  5. fig_auc_summary.pdf     —— 跨数据集 AUC 汇总条形图（含误差条）

输出：figures/*.pdf（矢量，内嵌字体）+ PNG 预览
============================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from datasets import DATASETS, DATA_DIR

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
    "figure.dpi": 100,
})

# 色盲友好调色板 (Okabe-Ito)
C = {
    "XGBoost": "#0072B2", "LightGBM": "#56B4E9", "CatBoost": "#009E73",
    "MLP": "#E69F00", "FT-Transformer": "#C44E52", "CNN-M1": "#D55E00",
    "CNN-M1c": "#CC79A7", "CNN-M2": "#8C8C8C", "CNN-M3-RG": "#BEBADA",
    "CNN-M3-full": "#000000", "CNN-M3-noG": "#F0E442",
    "CNN-M3-noB": "#66C2A5", "CNN-M3-corrB": "#A6D854", "CNN-M3-shapB": "#E78AC3",
    "CNN-IGTD": "#B2182B",
}

# 精选方法集（避免曲线过密）
PLOT_METHODS = ["XGBoost", "FT-Transformer", "MLP", "CNN-M1", "CNN-M1c",
                "CNN-M3-RG", "CNN-M3-full", "CNN-IGTD"]

FILE_MAP = {
    "XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cat", "MLP": "mlp",
    "FT-Transformer": "ft",
    "CNN-M1": "cnn_M1", "CNN-M1c": "cnn_M1c", "CNN-M2": "cnn_M2",
    "CNN-M3-full": "cnn_M3-full", "CNN-M3-noG": "cnn_M3-noG",
    "CNN-M3-noB": "cnn_M3-noB", "CNN-M3-corrB": "cnn_M3-corrB",
    "CNN-M3-shapB": "cnn_M3-shapB", "CNN-M3-RG": "cnn_M3-RG",
    "CNN-IGTD": "cnn_IGTD",
}
DATASET_LABEL = {"adult": "UCI Adult", "heart": "UCI Heart", "wine": "UCI Wine",
                 "bank": "UCI Bank", "credit": "UCI Credit",
                 "german": "UCI German", "telco": "Telco Churn", "sick": "UCI Sick",
                 "australian": "Austral. Credit", "cmc": "UCI CMC",
                 "ilpd": "ILPD", "segment": "Segmentation", "vehicle": "Vehicle",
                 "spambase": "Spambase", "magic": "MAGIC"}
ALL_DATASETS = ["adult", "heart", "wine", "bank", "credit", "german", "telco",
                "sick", "australian", "cmc", "ilpd", "segment", "vehicle",
                "spambase", "magic"]


def load(name, tag):
    p = os.path.join(DATA_DIR, f"{name}_{tag}_results.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["probs"], d["labels"]


def fig1_architecture():
    """M3 三通道编码 + CNN 端到端示意图（矢量）。"""
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

    # 左：表格行
    ax.add_patch(mpatches.FancyBboxPatch((0.15, 1.05), 1.9, 1.6,
                 boxstyle="round,pad=0.05", fc="#f7f7f7", ec="#333333", lw=1.0))
    ax.text(1.1, 2.75, "tabular row", ha="center", fontsize=9, style="italic")
    num_labs = ["age=38", "hours=42", "gain=2174", "loss=0"]
    cat_labs = ["workclass=Private", "marital=Married", "occ=Prof-spec"]
    for i, t in enumerate(num_labs):
        ax.add_patch(mpatches.FancyBboxPatch((0.3, 2.25 - i * 0.33), 0.85, 0.26,
                     boxstyle="round,pad=0.02", fc="#cce5ff", ec="#333333", lw=0.6))
        ax.text(0.42, 2.29 - i * 0.33, t, fontsize=6, ha="left", va="center")
    for i, t in enumerate(cat_labs):
        ax.add_patch(mpatches.FancyBboxPatch((1.25, 2.25 - i * 0.33), 0.72, 0.26,
                     boxstyle="round,pad=0.02", fc="#ffe6cc", ec="#333333", lw=0.6))
        ax.text(1.31, 2.29 - i * 0.33, t, fontsize=5.5, ha="left", va="center")

    # 箭头到编码
    ax.annotate("", xy=(2.55, 1.85), xytext=(2.1, 1.85),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"))
    ax.text(2.32, 2.0, "encoding\n$E_{\\mathrm{M3}}$", fontsize=8, ha="center")

    # 中：三通道
    ch_colors = {"R": "#e07b7b", "G": "#9bd19b", "B": "#7b9fe0"}
    ch_names = {"R": "R: numeric\nheatmap", "G": "G: categorical\none-hot", "B": "B: importance-\nreordered"}
    for i, (ch, col) in enumerate(ch_colors.items()):
        x0 = 3.0 + i * 2.0
        ax.add_patch(mpatches.FancyBboxPatch((x0, 0.55), 1.7, 2.1,
                     boxstyle="round,pad=0.04", fc=col, ec="#333333", lw=0.9))
        ax.text(x0 + 0.85, 2.35, ch, fontsize=13, ha="center", fontweight="bold",
                color="#111111")
        ax.text(x0 + 0.85, 0.75, ch_names[ch], fontsize=6.5, ha="center", va="bottom")
    ax.add_patch(mpatches.FancyBboxPatch((3.0, 0.55), 5.9, 2.1, boxstyle="round,pad=0.06",
                 fc="none", ec="#000000", lw=1.2, linestyle="--"))
    ax.text(5.95, 2.75, "M3 image  ($3\\times 16\\times 64$)", fontsize=8.5,
            ha="center", style="italic")

    # 箭头到 CNN
    ax.annotate("", xy=(9.35, 1.85), xytext=(8.95, 1.85),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"))

    # 右：CNN 块
    ax.add_patch(mpatches.FancyBboxPatch((9.4, 0.75), 0.55, 2.2,
                 boxstyle="round,pad=0.05", fc="#e6e6fa", ec="#333333", lw=1.0))
    ax.text(9.68, 1.9, "CNN\n$3{\\times}3$\nconv\nblocks\n\n\n", fontsize=7, ha="center", va="center")
    ax.text(9.68, 0.85, "→ pooled\n→ FC", fontsize=6, ha="center", va="center")
    ax.text(9.68, 2.7, "CNN classifier", fontsize=8, ha="center", style="italic")

    ax.set_title("Multi-channel image encoding (M3) as a probe for tabular deep learning",
                 fontsize=10, pad=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_architecture.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig1_architecture.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [fig1_architecture]")


def fig_roc_pr(name):
    """ROC 与 PR 曲线（精选方法集）。"""
    methods = {}
    for m in PLOT_METHODS:
        r = load(name, FILE_MAP[m])
        if r is not None:
            methods[m] = r
    y = next(iter(methods.values()))[1]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for m, (prob, _) in methods.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, color=C[m], lw=1.8, label=f"{m} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {DATASET_LABEL[name]}")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_roc_{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"fig_roc_{name}.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for m, (prob, _) in methods.items():
        prec, rec, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(rec, prec, color=C[m], lw=1.8, label=f"{m} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–recall — {DATASET_LABEL[name]}")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_pr_{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"fig_pr_{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig_roc_{name}] [fig_pr_{name}]")


def fig_encodings():
    """M1 / M1c / M3 编码示例对比（Adult 一个样本）。"""
    npz = np.load(os.path.join(DATA_DIR, "adult_arrays.npz"))
    Xn, Xc = npz["X_train_num"], npz["X_train_cat"]
    imp = np.load(os.path.join(DATA_DIR, "adult_xgb_importance.npz"))
    w_imp = imp["importance_numeric"]
    i = 0  # 任取一个训练样本

    def one_hot(vec):
        oh = np.zeros(int(vec.max()) + 1)
        oh[int(vec[0])] = 1.0
        return oh

    # M1
    m1 = np.tile(Xn[i], (64, 1)).T  # (6, 64)
    # M1c: 数值 6 行 + 类别 one-hot 行
    cat_rows = np.zeros((8, 64))
    for j in range(8):
        oh = one_hot(Xc[i][j:j + 1])
        cat_rows[j, :min(len(oh), 64)] = oh[:min(len(oh), 64)]
    m1c = np.vstack([m1, cat_rows])  # (14, 64)
    # M3: R=数值(填 16 行), G=one-hot(填 16 行), B=重排数值(填 16 行)
    def pad16(a):
        out = np.zeros((16, 64))
        out[:a.shape[0]] = a
        return out
    order = np.argsort(-w_imp)
    m3 = np.stack([pad16(m1), pad16(cat_rows), pad16(m1[order])], axis=0)  # (3,16,64)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, (title, img) in zip(axes, [
        ("M1 — numeric grayscale", m1[None, :, :]),
        ("M1c — single channel + categorical", m1c[None, :, :]),
        ("M3 — RGB (R/G/B)", m3),
    ]):
        if img.shape[0] == 3:
            ax.imshow(np.transpose(img, (1, 2, 0)), aspect="auto", vmin=0, vmax=1)
        else:
            ax.imshow(img[0], cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Example encodings of one UCI Adult row", fontsize=10, y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_encodings.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_encodings.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [fig_encodings]")


def fig_auc_summary():
    """跨資料集 AUC 熱圖（方法 × 資料集，格=mean AUC over seeds）。"""
    order = ["XGBoost", "FT-Transformer", "MLP", "CNN-M1", "CNN-M1c",
             "CNN-M3-RG", "CNN-M3-full"]
    ds_list = ALL_DATASETS
    mat = np.full((len(ds_list), len(order)), np.nan)
    for i, name in enumerate(ds_list):
        for j, m in enumerate(order):
            r = load(name, FILE_MAP[m])
            if r is None:
                continue
            psp = np.load(os.path.join(DATA_DIR, f"{name}_{FILE_MAP[m]}_results.npz"))["per_seed_probs"]
            y = r[1]
            aucs = [roc_auc_score(y, psp[s]) for s in range(psp.shape[0])]
            mat[i, j] = np.mean(aucs)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    cmap = matplotlib.colormaps["YlGnBu"] if hasattr(matplotlib, "colormaps") else matplotlib.cm.YlGnBu
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0.55, vmax=1.0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ds_list)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in ds_list], fontsize=8)
    for i in range(len(ds_list)):
        for j in range(len(order)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v < 0.82 else "#222222")
    ax.set_title("Test-set AUC: methods (columns) × datasets (rows)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_auc_summary.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_auc_summary.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [fig_auc_summary]")


def main():
    fig1_architecture()
    for name in ALL_DATASETS:
        fig_roc_pr(name)
    fig_encodings()
    fig_auc_summary()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
