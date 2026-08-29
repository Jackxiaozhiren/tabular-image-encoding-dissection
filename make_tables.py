"""
make_tables.py
============================================================
從 data/*.npz 與 *.json 自動生成稿件表格的 LaTeX 行（mean±std）。

用法：python3 make_tables.py > /tmp/tables_out.txt
（需先跑完 01/03/07/04/10/08；輸出貼入 manuscript.tex）
============================================================
"""
import json
import os
import numpy as np

from datasets import DATASETS, DATA_DIR

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
DATASETS_LIST = ["adult", "heart", "wine", "bank", "credit", "german", "telco",
                 "sick", "australian", "cmc", "ilpd", "segment", "vehicle",
                 "spambase", "magic"]
FILE_MAP = {
    "XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cat", "MLP": "mlp",
    "FT-Transformer": "ft",
    "CNN-M1": "cnn_M1", "CNN-M1c": "cnn_M1c", "CNN-M2": "cnn_M2",
    "CNN-M3-full": "cnn_M3-full", "CNN-M3-noG": "cnn_M3-noG",
    "CNN-M3-noB": "cnn_M3-noB", "CNN-M3-corrB": "cnn_M3-corrB",
    "CNN-M3-shapB": "cnn_M3-shapB", "CNN-M3-RG": "cnn_M3-RG",
    "CNN-IGTD": "cnn_IGTD",
}
ORDER = ["XGBoost", "LightGBM", "CatBoost", "FT-Transformer", "MLP",
         "CNN-M1", "CNN-M1c", "CNN-M2", "CNN-M3-RG", "CNN-M3-full",
         "CNN-M3-noG", "CNN-M3-noB", "CNN-M3-corrB", "CNN-M3-shapB",
         "CNN-IGTD"]
# 各資料集可用方法（Wine 全數值：M1c/M3-noG 退化不訓練）
def methods_for(ds):
    excl = {"wine": {"CNN-M1c", "CNN-M3-noG"}} if ds == "wine" else set()
    return [m for m in ORDER if m not in excl]


def load(name, tag):
    p = os.path.join(DATA_DIR, f"{name}_{tag}_results.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["per_seed_metrics"], d["metric_names"]


def fmt_row(ds, m):
    r = load(ds, FILE_MAP[m])
    if r is None:
        return None
    psm, names = r
    cells = [m if m != "CNN-M3-full" else "CNN-M3-full (probe)"]
    for k in range(5):
        v = psm[:, k].mean(); s = psm[:, k].std()
        cells.append(f"{v:.4f}$\\pm${s:.4f}")
    return " & ".join(cells) + " \\\\"


def main():
    # ---------- 資料集表（表 1） ----------
    ds = json.load(open(os.path.join(DATA_DIR, "dataset_summary.json")))
    print("===== TABLE 1: DATASETS =====")
    for name in DATASETS_LIST:
        d = ds[name]
        ptr = d["pos_train"] / (d["pos_train"] + d["neg_train"]) * 100
        pte = d["pos_test"] / (d["pos_test"] + d["neg_test"]) * 100
        label = {"adult": "Census", "heart": "Medical", "wine": "Chemical",
                 "bank": "Financial", "credit": "Financial", "german": "Financial",
                 "telco": "Telecom", "sick": "Medical", "australian": "Financial",
                 "cmc": "Demographic", "ilpd": "Medical", "segment": "Image",
                 "vehicle": "Image", "spambase": "Text", "magic": "Physics"}[name]
        split = "Official" if name == "adult" else "Strat. 75/25"
        disp = DATASETS[name]["display"].replace("UCI ", "").replace(" (", " (").replace(" Statlog", "").replace(" (Census Income)", " (Census Income)")
        print(f"{disp} & {d['train']:,} & {d['test']:,} & {d['numeric']} & {d['categorical']} & {ptr:.1f}/{pte:.1f} & {label} & {split} \\\\")

    # ---------- 每資料集效能表 ----------
    print("\n===== PER-DATASET TABLES =====")
    for ds in DATASETS_LIST:
        print(f"\n--- {ds} ---")
        for m in methods_for(ds):
            row = fmt_row(ds, m)
            if row:
                print(row)

    # ---------- 顯著性表（表 5） ----------
    sig = json.load(open(os.path.join(DATA_DIR, "extended_significance.json")))
    print("\n===== SIGNIFICANCE (vs M3-full) =====")
    def fmt_p(p):
        return "$<0.0001$" if p < 0.0001 else f"${p:.4f}$"
    for ds in DATASETS_LIST:
        if ds == "heart":
            continue
        for m, v in sig.get(ds, {}).items():
            if m == "CNN-M3-full":
                continue
            print(f"{ds} & {m} vs. M3-full & {fmt_p(v['mcnemar_p'])} & {fmt_p(v['delong_p'])} \\\\")

    # ---------- Wilcoxon 表（表 6） ----------
    w = json.load(open(os.path.join(DATA_DIR, "extended_wilcoxon.json")))
    print("\n===== WILCOXON =====")
    for p in w["pairs"]:
        print(f"{p['a']} vs {p['b']} & {p['n']} & ${p['median_delta']:+.4f}$ & ${p['p']:.4f}$ \\\\")
        print(f"   excl-Heart n={p['n_excl_heart']} p={p['p_excl_heart']:.4f} | ds-median p={p['p_dataset_median']:.4f} | holm={p['holm_reject']}")


if __name__ == "__main__":
    main()
