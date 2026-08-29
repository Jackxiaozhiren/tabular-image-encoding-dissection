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
DATASETS_LIST = ["adult", "heart", "wine", "bank", "credit", "german", "telco",
                 "sick", "australian", "cmc", "ilpd", "segment", "vehicle",
                 "spambase", "magic"]

# 全數值資料集（無類別特徵）：M1c ≡ M1、M3-noG ≡ M3-full，故不訓練
NUMERIC_ONLY = {"wine", "segment", "vehicle", "spambase", "magic"}

# 全部方法
ALL_METHODS = ["XGBoost", "LightGBM", "CatBoost", "MLP", "FT-Transformer",
               "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-full",
               "CNN-M3-noG", "CNN-M3-noB", "CNN-M3-corrB", "CNN-M3-shapB",
               "CNN-M3-RG", "CNN-IGTD"]
METHODS = {
    ds: [m for m in ALL_METHODS
         if not (ds in NUMERIC_ONLY and m in ("CNN-M1c", "CNN-M3-noG"))]
    for ds in DATASETS_LIST
}

FILE_MAP = {
    "XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cat", "MLP": "mlp",
    "FT-Transformer": "ft",
    "CNN-M1": "cnn_M1", "CNN-M1c": "cnn_M1c", "CNN-M2": "cnn_M2",
    "CNN-M3-full": "cnn_M3-full", "CNN-M3-noG": "cnn_M3-noG",
    "CNN-M3-noB": "cnn_M3-noB", "CNN-M3-corrB": "cnn_M3-corrB",
    "CNN-M3-shapB": "cnn_M3-shapB", "CNN-M3-RG": "cnn_M3-RG",
    "CNN-IGTD": "cnn_IGTD",
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
    per_seed_auc = {m: {} for m in ALL_METHODS}
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
    print("\n===== 跨数据集 Wilcoxon signed-rank（pooled seed×dataset） =====")
    # 預設配對家族（10 對照）：類別納入 / 通道分離 / B 通道 / 重排 / 權重來源 /
    # 影像形式 / 強力表格 DL / 樹 vs 影像
    pairs = [("CNN-M1", "CNN-M3-full"),       # 類別納入（複合對照）
             ("CNN-M3-noG", "CNN-M3-full"),   # 類別通道（乾淨）
             ("CNN-M1c", "CNN-M3-RG"),        # 通道分離（乾淨，B 置零）
             ("CNN-M3-RG", "CNN-M3-full"),    # B 通道有無
             ("CNN-M3-noB", "CNN-M3-full"),   # 重排
             ("CNN-M3-corrB", "CNN-M3-full"), # 權重來源（線性）
             ("CNN-M3-shapB", "CNN-M3-full"), # 權重來源（SHAP）
             ("MLP", "CNN-M3-full"),          # 影像形式
             ("FT-Transformer", "MLP"),       # 強力表格 DL vs MLP 對照
             ("XGBoost", "CNN-M3-full")]      # 樹 vs 影像

    def _wp(d):
        """双侧 signed-rank；处理全零与空数组退化情形。"""
        if len(d) == 0 or np.all(d == 0):
            return 1.0
        return float(stats.wilcoxon(d, alternative="two-sided").pvalue)

    # 收集每个配对的 (dataset, seed) 差异，供敏感性重算
    pair_diffs = {}
    for a, b in pairs:
        va = np.array([per_seed_auc[a][k] for k in per_seed_auc[a]
                       if k in per_seed_auc[b]])
        if len(va) == 0:
            continue
        keys = [k for k in per_seed_auc[a] if k in per_seed_auc[b]]
        pair_diffs[(a, b)] = {k: (per_seed_auc[a][k] - per_seed_auc[b][k])
                              for k in keys}

    def _excl_heart(diffs):
        return np.array([v for k, v in diffs.items() if k[0] != "heart"])

    def _ds_medians(diffs):
        by_ds = {}
        for (ds, s), v in diffs.items():
            by_ds.setdefault(ds, []).append(v)
        return np.array([np.median(v) for v in by_ds.values()])

    wilcox = []
    for (a, b), diffs in pair_diffs.items():
        diffs_all = np.array(list(diffs.values()))
        h_excl = _excl_heart(diffs)
        dmed = _ds_medians(diffs)
        wilcox.append({
            "a": a, "b": b, "n": len(diffs_all),
            "median_delta": float(np.median(diffs_all)),
            "p": _wp(diffs_all), "mean_delta": float(diffs_all.mean()),
            "n_excl_heart": len(h_excl), "p_excl_heart": _wp(h_excl),
            "n_datasets": len(dmed), "p_dataset_median": _wp(dmed),
        })
        print(f"  {a:>14s} vs {b:14s}  n={len(diffs_all)}  "
              f"median ΔAUC={np.median(diffs_all):+.4f}  p={_wp(diffs_all):.4f}")
    # Holm-Bonferroni 校正（对配对集合）
    pvals = np.array([w["p"] for w in wilcox])
    reject = holm_bonferroni(pvals)
    for w, rj in zip(wilcox, reject):
        w["holm_reject"] = rj
        print(f"    Holm-Bonferroni 拒绝: {rj}")
    with open(os.path.join(DATA_DIR, "extended_wilcoxon.json"), "w") as f:
        json.dump({"pairs": wilcox, "datasets": DATASETS_LIST,
                   "n_seeds": len(SEEDS),
                   "note": "pooled seed×dataset paired AUC (+sensitivity: excl-Heart, dataset-median)"},
                  f, ensure_ascii=False, indent=1)


def compute_metrics(y, yp, prob):
    return {"Accuracy": accuracy_score(y, yp), "Precision": precision_score(y, yp),
            "Recall": recall_score(y, yp), "F1": f1_score(y, yp),
            "AUC": roc_auc_score(y, prob)}


if __name__ == "__main__":
    main()
