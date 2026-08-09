"""
06_ordering_analysis.py
============================================================
權重來源分歧診斷

  比較同一資料集上三種權重（XGBoost 重要度、SHAP 平均 |貢獻|、
  與標籤的線性相關係數）對「數值特徵」的排序是否一致。
  用於解釋權重重排/權重來源消融的結果：
    若三者排序高度一致，則「用哪個權重來源」自然對效能幾乎無影響；
    若分歧大（如高維、非線性關係主導的資料），權重重排的價值可能更高。

執行：python3 06_ordering_analysis.py [--datasets adult,wine]
輸出：終端表格 + data/ordering_divergence.json
============================================================
"""
import argparse
import json
import os
import numpy as np
from scipy import stats

from datasets import DATASETS, DATA_DIR


def analyze(name):
    cfg = DATASETS[name]
    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    X_train_num, y_train = npz["X_train_num"], npz["y_train"]
    imp = np.load(os.path.join(DATA_DIR, f"{name}_xgb_importance.npz"))
    n_num = len(cfg["numeric"])

    w_imp = imp["importance_numeric"][:n_num]
    w_shap = imp["shap_importance_numeric"][:n_num]
    w_corr = np.array([np.corrcoef(X_train_num[:, k], y_train)[0, 1]
                       for k in range(X_train_num.shape[1])])

    # 取絕對值（SHAP 與相關係數有正負方向）
    w_imp_a, w_shap_a, w_corr_a = np.abs(w_imp), np.abs(w_shap), np.abs(w_corr)

    # 排序（降序）
    order_imp = np.argsort(-w_imp_a)
    order_shap = np.argsort(-w_shap_a)
    order_corr = np.argsort(-w_corr_a)

    # 成對秩相關（Kendall tau 與 Spearman）
    def rank_corr(o1, o2):
        r1 = np.empty_like(o1); r1[o1] = np.arange(len(o1))
        r2 = np.empty_like(o2); r2[o2] = np.arange(len(o2))
        tau = stats.kendalltau(r1, r2).statistic
        rho = stats.spearmanr(r1, r2).statistic
        return tau, rho

    pairs = {
        "importance_vs_corr": rank_corr(order_imp, order_corr),
        "importance_vs_shap": rank_corr(order_imp, order_shap),
        "corr_vs_shap": rank_corr(order_corr, order_shap),
    }

    print(f"\n===== {cfg['display']}（{n_num} 個數值特徵） =====")
    print("  數值特徵排序（降序）：")
    print(f"    {'特徵':24s} {'XGBoost':>10s} {'SHAP':>10s} {'|corr|':>10s}")
    for i in range(n_num):
        f = cfg["numeric"][i]
        print(f"    {f:24s} {w_imp_a[i]:10.4f} {w_shap_a[i]:10.4f} "
              f"{w_corr_a[i]:10.4f}")
    print("  成對秩相關（Kendall tau / Spearman）：")
    for k, (tau, rho) in pairs.items():
        print(f"    {k:24s} tau={tau:.3f}  rho={rho:.3f}")

    return {"dataset": name, "n_numeric": n_num, "rank_corr": pairs,
            "order_importance": [cfg["numeric"][i] for i in order_imp],
            "order_shap": [cfg["numeric"][i] for i in order_shap],
            "order_corr": [cfg["numeric"][i] for i in order_corr]}


def main():
    ap = argparse.ArgumentParser(description="權重來源分歧診斷")
    ap.add_argument("--datasets", default="adult,wine")
    args = ap.parse_args()
    out = {}
    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        if name in DATASETS:
            out[name] = analyze(name)
        else:
            print(f"[警告] 忽略未知資料集: {name}")
    with open(os.path.join(DATA_DIR, "ordering_divergence.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已儲存 data/ordering_divergence.json")


if __name__ == "__main__":
    main()
