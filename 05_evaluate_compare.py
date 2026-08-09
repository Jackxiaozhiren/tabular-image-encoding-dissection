"""
05_evaluate_compare.py
============================================================
實驗結果與分析：整合 XGBoost 與各 CNN 編碼/消融變體的結果

  1. 讀取 03/04 儲存的預測結果（含多次種子），重算各指標 mean ± std
  2. 輸出對比表格（Markdown 格式，可直接貼入論文）
  3. 統計顯著性檢定：
     - McNemar 檢定（0.5 閾值之 Accuracy 差異）
     - DeLong 檢定（AUC 差異），與 M3-full 兩兩比較
  4. 繪製 ROC / PR 曲線比較圖

執行：python3 05_evaluate_compare.py --dataset adult [--variants full,noG,noB,corrB,shapB]
輸出：figures/fig_roc_compare_{name}.png、fig_pr_compare_{name}.png、
      data/significance_{name}.json
============================================================
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "PingFang SC",
                                   "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)

from datasets import DATASETS, DATA_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
COLORS = {"XGBoost": "#d62728", "CNN-M1": "#1f77b4",
          "CNN-M2": "#2ca02c", "CNN-M3-full": "#9467bd",
          "CNN-M3-noG": "#ff7f0e", "CNN-M3-noB": "#8c564b",
          "CNN-M3-corrB": "#17becf", "CNN-M3-shapB": "#e377c2"}


def load_methods(name, variants):
    """載入各方法之 (mean_probs, per_seed_probs, labels)。"""
    methods = {}
    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    y_test = npz["y_test"]

    xgb = np.load(os.path.join(DATA_DIR, f"{name}_xgb_results.npz"))
    methods["XGBoost"] = (xgb["probs"], xgb["per_seed_probs"], y_test)
    for mode in ["M1", "M1c", "M2"]:
        p = os.path.join(DATA_DIR, f"{name}_cnn_{mode}_results.npz")
        if os.path.exists(p):
            c = np.load(p)
            methods[f"CNN-{mode}"] = (c["probs"], c["per_seed_probs"], c["labels"])
    for v in variants:
        c = np.load(os.path.join(DATA_DIR, f"{name}_cnn_M3-{v}_results.npz"))
        methods[f"CNN-M3-{v}"] = (c["probs"], c["per_seed_probs"], c["labels"])
    return methods


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_prob),
    }


# ------------------------- 統計檢定 -------------------------
def mcnemar_test(y_true, p_a, p_b):
    """McNemar 檢定：兩方法 0.5 閾值預測的準確率是否顯著不同。回傳 p 值。"""
    pa, pb = (p_a >= 0.5).astype(int), (p_b >= 0.5).astype(int)
    b = int(((pa == y_true) & (pb != y_true)).sum())   # A 對、B 錯
    c = int(((pa != y_true) & (pb == y_true)).sum())   # B 對、A 錯
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)             # 連續性校正
    return float(1 - stats.chi2.cdf(stat, 1))


def _per_class_auc_stats(y_true, prob):
    """DeLong 演算法所需的逐樣本統計量（V10、V01）。"""
    y = np.asarray(y_true); p = np.asarray(prob)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    p_pos, p_neg = p[y == 1], p[y == 0]
    # V10[i]：正例 i 勝過負例的比例（含平手 0.5）
    sorted_neg = np.sort(p_neg)
    less = np.searchsorted(sorted_neg, p_pos, side="left")
    le = np.searchsorted(sorted_neg, p_pos, side="right")
    V10 = (less + 0.5 * (le - less)) / n_neg
    # V01[j]：負例 j 被正例勝過的比例（含平手 0.5）
    sorted_pos = np.sort(p_pos)
    ge = n_pos - np.searchsorted(sorted_pos, p_neg, side="left")   # >= p_j
    gt = n_pos - np.searchsorted(sorted_pos, p_neg, side="right")  # >  p_j
    V01 = (gt + 0.5 * (ge - gt)) / n_pos
    return V10, V01, n_pos, n_neg


def delong_test(y_true, p_a, p_b):
    """DeLong 檢定：兩模型的 AUC 是否顯著不同。回傳 p 值。"""
    v10a, v01a, n1, n0 = _per_class_auc_stats(y_true, p_a)
    v10b, v01b, _, _ = _per_class_auc_stats(y_true, p_b)
    auc_a, auc_b = v10a.mean(), v10b.mean()
    var_a = v10a.var(ddof=1) / n1 + v01a.var(ddof=1) / n0
    var_b = v10b.var(ddof=1) / n1 + v01b.var(ddof=1) / n0
    cov10 = np.cov(v10a, v10b, ddof=1)[0, 1] / n1
    cov01 = np.cov(v01a, v01b, ddof=1)[0, 1] / n0
    var_diff = var_a + var_b - 2 * (cov10 + cov01)
    if var_diff <= 0:
        return 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser(description="彙總、比較並檢定各方法")
    ap.add_argument("--dataset", default="adult", choices=list(DATASETS.keys()))
    ap.add_argument("--variants", default="full",
                    help="逗號分隔的 M3 消融變體（須先以 04 訓練）")
    args = ap.parse_args()
    name, cfg = args.dataset, DATASETS[args.dataset]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    methods = load_methods(name, variants)

    # ---------- 指標 mean ± std（跨種子） ----------
    print(f"===== 方法對比表（{cfg['display']}，mean ± std，n=3 種子） =====")
    header = ["Method"] + METRIC_NAMES
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for mname, (mean_prob, per_seed, y_true) in methods.items():
        per_seed_metrics = [compute_metrics(y_true, (p >= 0.5).astype(int), p)
                            for p in per_seed]
        cells = [mname]
        for k in METRIC_NAMES:
            vals = np.array([m[k] for m in per_seed_metrics])
            cells.append(f"{vals.mean():.4f} ± {vals.std():.4f}")
        print("| " + " | ".join(cells) + " |")

    # ---------- 統計顯著性（與 M3-full 比較） ----------
    if "CNN-M3-full" in methods:
        ref = methods["CNN-M3-full"]
        print(f"\n===== 統計檢定（基準：CNN-M3-full，n_test={len(ref[2])}） =====")
        print("| 比較 | ΔAccuracy | McNemar p | ΔAUC | DeLong p |")
        print("|---|---|---|---|---|")
        sig = {}
        for mname, (mean_prob, _, y_true) in methods.items():
            if mname == "CNN-M3-full":
                continue
            p_mc = mcnemar_test(y_true, mean_prob, ref[0])
            p_dl = delong_test(y_true, mean_prob, ref[0])
            acc_diff = accuracy_score(y_true, (mean_prob >= 0.5).astype(int)) \
                - accuracy_score(y_true, (ref[0] >= 0.5).astype(int))
            auc_diff = roc_auc_score(y_true, mean_prob) - roc_auc_score(y_true, ref[0])
            sig[mname] = {"mcnemar_p": p_mc, "delong_p": p_dl,
                          "acc_diff": acc_diff, "auc_diff": auc_diff}
            print(f"| {mname} vs M3-full | {acc_diff:+.4f} | {p_mc:.4f} "
                  f"| {auc_diff:+.4f} | {p_dl:.4f} |")
        with open(os.path.join(DATA_DIR, f"significance_{name}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sig, f, ensure_ascii=False, indent=2)

    # ---------- ROC 曲線 ----------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for mname, (mean_prob, _, y_true) in methods.items():
        fpr, tpr, _ = roc_curve(y_true, mean_prob)
        auc = roc_auc_score(y_true, mean_prob)
        ax.plot(fpr, tpr, color=COLORS.get(mname, "#333333"), lw=2,
                label=f"{mname} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="隨機猜測")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC 曲線比較（{cfg['display']}）")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_roc_compare_{name}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"\n已儲存 fig_roc_compare_{name}.png")

    # ---------- PR 曲線 ----------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for mname, (mean_prob, _, y_true) in methods.items():
        prec, rec, _ = precision_recall_curve(y_true, mean_prob)
        ap = average_precision_score(y_true, mean_prob)
        ax.plot(rec, prec, color=COLORS.get(mname, "#333333"), lw=2,
                label=f"{mname} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall 曲線比較（{cfg['display']}）")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_pr_compare_{name}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"已儲存 fig_pr_compare_{name}.png")


if __name__ == "__main__":
    main()
