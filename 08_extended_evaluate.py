"""
08_extended_evaluate.py
============================================================
阶段三：扩展评估——现代基线与决定性对照的完整统计检验

  1. 每数据集全方法对比表（mean ± std over 3 seeds）
  2. McNemar / DeLong 检验（seed-平均预测，与 05 口径一致）
  3. 跨数据集 Wilcoxon signed-rank + Holm-Bonferroni 多重比较校正
     （pooled seed×dataset 配对 AUC，n = 3 数据集 × 3 种子 = 9）

输出：
  data/extended_significance.json   （每数据集 McNemar/DeLong）
  data/extended_wilcoxon.json       （跨数据集 Wilcoxon + Holm）
  终端对比表
============================================================
"""
import json
import os
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from datasets import DATASETS, DATA_DIR

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
SEEDS = [42, 123, 2024]
DATASETS_LIST = ["adult", "heart", "wine", "bank", "credit"]

# 每数据集的方法清单（含 M3 消融；heart 因小样本只跑 full）
METHODS = {
    "adult": ["XGBoost", "LightGBM", "CatBoost", "MLP",
              "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
              "CNN-M3-noG", "CNN-M3-noB", "CNN-M3-corrB", "CNN-M3-shapB"],
    "heart": ["XGBoost", "LightGBM", "CatBoost", "MLP",
              "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
              "CNN-M3-noG", "CNN-M3-noB"],
    "wine": ["XGBoost", "LightGBM", "CatBoost", "MLP",
             "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
             "CNN-M3-noB", "CNN-M3-corrB", "CNN-M3-shapB"],
    "bank": ["XGBoost", "LightGBM", "CatBoost", "MLP",
             "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
             "CNN-M3-noG", "CNN-M3-noB"],
    "credit": ["XGBoost", "LightGBM", "CatBoost", "MLP",
               "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
               "CNN-M3-noG", "CNN-M3-noB"],
}

FILE_MAP = {
    "XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cat", "MLP": "mlp",
    "CNN-M1": "cnn_M1", "CNN-M1c": "cnn_M1c", "CNN-M2": "cnn_M2",
    "CNN-M3-full": "cnn_M3-full", "CNN-M3-noG": "cnn_M3-noG",
    "CNN-M3-noB": "cnn_M3-noB", "CNN-M3-corrB": "cnn_M3-corrB",
    "CNN-M3-shapB": "cnn_M3-shapB",
}


def load_method(name, tag):
    p = os.path.join(DATA_DIR, f"{name}_{tag}_results.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {"probs": d["probs"], "labels": d["labels"],
            "per_seed_probs": d["per_seed_probs"]}


def mcnemar_p(y, pa, pb):
    a = ((pa >= 0.5).astype(int) == y) & ((pb >= 0.5).astype(int) != y)
    b = ((pa >= 0.5).astype(int) != y) & ((pb >= 0.5).astype(int) == y)
    na, nb = int(a.sum()), int(b.sum())
    if na + nb == 0:
        return 1.0
    return float(1 - stats.chi2.cdf((abs(na - nb) - 1) ** 2 / (na + nb), 1))


def delong_p(y, pa, pb):
    def vstats(p):
        pos, neg = p[y == 1], p[y == 0]
        sn = np.sort(neg)
        less = np.searchsorted(sn, pos, side="left")
        le = np.searchsorted(sn, pos, side="right")
        v10 = (less + 0.5 * (le - less)) / len(neg)
        sp = np.sort(pos)
        ge = len(pos) - np.searchsorted(sp, neg, side="left")
        gt = len(pos) - np.searchsorted(sp, neg, side="right")
        v01 = (gt + 0.5 * (ge - gt)) / len(pos)
        return v10, v01
    v10a, v01a = vstats(pa); v10b, v01b = vstats(pb)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    va = v10a.var(ddof=1) / n1 + v01a.var(ddof=1) / n0
    vb = v10b.var(ddof=1) / n1 + v01b.var(ddof=1) / n0
    cov = np.cov(v10a, v10b, ddof=1)[0, 1] / n1 + np.cov(v01a, v01b, ddof=1)[0, 1] / n0
    vd = va + vb - 2 * cov
    if vd <= 0:
        return 1.0
    return float(2 * (1 - stats.norm.cdf(abs((v10a.mean() - v10b.mean()) / np.sqrt(vd)))))


def holm_bonferroni(pvals):
    """Holm-Bonferroni 校正（降序 p 值，逐步比较 alpha/(m-k+1)）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for k, i in enumerate(order):
        if pvals[i] <= 0.05 / (m - k):
            reject[i] = True
        else:
            # 一旦不拒绝，其后的（更大 p 值）也不拒绝
            break
    return reject


def main():
    # ---------- 1. 每数据集对比表 ----------
    per_seed_auc = {m: {} for m in ["XGBoost", "LightGBM", "CatBoost", "MLP",
                                    "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
                                    "CNN-M3-noG", "CNN-M3-noB",
                                    "CNN-M3-corrB", "CNN-M3-shapB"]}
    sig_out = {}
    for name in DATASETS_LIST:
        print(f"\n===== {DATASETS[name]['display']} =====")
        methods = {}
        for m in METHODS[name]:
            r = load_method(name, FILE_MAP[m])
            if r is not None:
                methods[m] = r
        if "CNN-M3-full" not in methods:
            print("  [skip] CNN-M3-full not trained yet")
            continue
        ref = methods["CNN-M3-full"]
        y = ref["labels"]
        header = ["Method"] + METRIC_NAMES
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        ds_sig = {}
        for m, r in methods.items():
            psm = np.array([mcnemar_p(y, r["per_seed_probs"][s],
                                      ref["per_seed_probs"][s]) for s in range(3)])
            per_seed = [compute_metrics(y, (r["per_seed_probs"][s] >= 0.5).astype(int),
                                        r["per_seed_probs"][s]) for s in range(3)]
            cells = [m]
            for k in METRIC_NAMES:
                vals = np.array([x[k] for x in per_seed])
                cells.append(f"{vals.mean():.4f}±{vals.std():.4f}")
            print("| " + " | ".join(cells) + " |")
            # 记录 per-seed AUC（供跨数据集 Wilcoxon）
            for s in range(3):
                per_seed_auc[m][(name, s)] = per_seed[s]["AUC"]
            if m != "CNN-M3-full":
                p_mc = mcnemar_p(y, r["probs"], ref["probs"])
                p_dl = delong_p(y, r["probs"], ref["probs"])
                ds_sig[m] = {"mcnemar_p": p_mc, "delong_p": p_dl}
        sig_out[name] = ds_sig
    with open(os.path.join(DATA_DIR, "extended_significance.json"), "w") as f:
        json.dump(sig_out, f, ensure_ascii=False, indent=1)

    # ---------- 2. 跨数据集 Wilcoxon signed-rank + Holm ----------
    print("\n===== 跨数据集 Wilcoxon signed-rank（pooled seed×dataset，n=9） =====")
    # 关键配对：相对 M3-full；另有 MLP 决定性对照
    pairs = [("CNN-M1", "CNN-M3-full"), ("CNN-M1c", "CNN-M3-full"),
             ("MLP", "CNN-M3-full"), ("XGBoost", "CNN-M3-full"),
             ("CNN-M3-noB", "CNN-M3-full"), ("CNN-M1c", "MLP")]
    wilcox = []
    for a, b in pairs:
        va = np.array([per_seed_auc[a][k] for k in per_seed_auc[a]
                       if k in per_seed_auc[b]])
        vb = np.array([per_seed_auc[b][k] for k in per_seed_auc[a]
                       if k in per_seed_auc[b]])
        if len(va) == 0 or len(vb) == 0:
            continue
        diffs = va - vb
        # 双侧 signed-rank；处理全部为零的退化情形
        if np.all(diffs == 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
        wmed = float(np.median(diffs))
        wilcox.append({"a": a, "b": b, "n": len(va),
                       "median_delta": wmed, "p": p,
                       "mean_delta": float(diffs.mean())})
        print(f"  {a:>14s} vs {b:14s}  n={len(va)}  "
              f"median ΔAUC={wmed:+.4f}  p={p:.4f}")
    # Holm-Bonferroni 校正（对配对集合）
    pvals = np.array([w["p"] for w in wilcox])
    reject = holm_bonferroni(pvals)
    for w, rj in zip(wilcox, reject):
        w["holm_reject"] = rj
        print(f"    Holm-Bonferroni 拒绝: {rj}")
    with open(os.path.join(DATA_DIR, "extended_wilcoxon.json"), "w") as f:
        json.dump({"pairs": wilcox, "datasets": DATASETS_LIST,
                   "n_seeds": len(SEEDS), "note": "pooled seed×dataset paired AUC"},
                  f, ensure_ascii=False, indent=1)


def compute_metrics(y, yp, prob):
    return {"Accuracy": accuracy_score(y, yp), "Precision": precision_score(y, yp),
            "Recall": recall_score(y, yp), "F1": f1_score(y, yp),
            "AUC": roc_auc_score(y, prob)}


if __name__ == "__main__":
    main()
