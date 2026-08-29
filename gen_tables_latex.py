#!/usr/bin/env python3
"""Generate full LaTeX for the new manuscript tables (8 datasets). Output to stdout."""
import json
import os
import numpy as np
from datasets import DATASETS, DATA_DIR

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
DATASETS_LIST = ["adult", "heart", "wine", "bank", "credit", "german", "telco", "sick"]
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
DISPLAY = {
    "adult": "UCI Adult", "heart": "UCI Heart", "wine": "UCI Wine",
    "bank": "UCI Bank", "credit": "UCI Credit", "german": "UCI German",
    "telco": "Telco Churn", "sick": "UCI Sick"}

def methods_for(ds):
    excl = {"wine": {"CNN-M1c", "CNN-M3-noG"}}
    return [m for m in ORDER if m not in excl.get(ds, set())]

def load(ds, tag):
    p = os.path.join(DATA_DIR, f"{ds}_{tag}_results.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["per_seed_metrics"]

def table_rows(ds):
    rows = []
    for m in methods_for(ds):
        psm = load(ds, FILE_MAP[m])
        if psm is None:
            continue
        label = "CNN-M3-full (probe)" if m == "CNN-M3-full" else m
        cells = [label]
        for k in range(5):
            v = f"{psm[:, k].mean():.4f}$\\pm${psm[:, k].std():.4f}"
            if m == "CNN-M3-full" and k == 4:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        rows.append(" & ".join(cells) + " \\\\")
    return rows

CAPTION = {
    "adult": "Test-set performance on UCI Adult, including GBDT baselines, the FT-Transformer and MLP tabular baselines, baseline encodings, and M3 ablation variants. The reference encoding M3-full (probe) is shown in bold.",
    "heart": "Test-set performance on UCI Heart (class-balanced training; only 76 test samples). Trees, FT-Transformer, and the MLP control are clearly best; most image-encoding CNNs collapse toward all-positive predictions (recall~$1.0$), while the numeric-only M1 and IGTD encodings degrade to unreliable recall. The reference encoding M3-full is shown in bold.",
    "wine": "Test-set performance on UCI Wine (fully numeric; isolates reordering and the image form because there are no categorical features). The reference encoding M3-full is shown in bold.",
    "bank": "Test-set performance on UCI Bank Marketing (7 numeric, 9 categorical features). The reference encoding M3-full is shown in bold.",
    "credit": "Test-set performance on UCI Credit Card Default (22 numeric, 1 categorical feature). The reference encoding M3-full is shown in bold.",
    "german": "Test-set performance on UCI German Credit (7 numeric, 13 categorical features), an additional mixed-type dataset with many categorical features. The reference encoding M3-full is shown in bold.",
    "telco": "Test-set performance on Telco Customer Churn (4 numeric, 15 categorical features), an additional mixed-type dataset. The reference encoding M3-full is shown in bold.",
    "sick": "Test-set performance on UCI Sick (Thyroid; 6 numeric, 21 categorical features), an additional mixed-type dataset. The reference encoding M3-full is shown in bold.",
}

for ds in DATASETS_LIST:
    print(f"\\begin{{table}}[t]")
    print("\\centering")
    print("\\footnotesize\\setlength{\\tabcolsep}{3pt}")
    print(f"\\caption{{{CAPTION[ds]}}}")
    print(f"\\label{{tab:{ds}}}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Method & Accuracy & Precision & Recall & F1 & AUC \\\\")
    print("\\midrule")
    for r in table_rows(ds):
        print(r)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{table}")
    print()

# Significance table (vs M3-full)
sig = json.load(open(os.path.join(DATA_DIR, "extended_significance.json")))
print("\\begin{table}[t]")
print("\\footnotesize\\setlength{\\tabcolsep}{3pt}")
print("\\centering")
print("\\caption{Per-dataset statistical tests versus the reference CNN-M3-full. McNemar $p$ for $0.5$-threshold accuracy; DeLong $p$ for AUC, both on seed-averaged predictions. Shown are the comparisons discussed in the text; Heart is excluded (76 test samples, degenerate CNNs).}")
print("\\label{tab:significance}")
print("\\begin{tabular}{llcc}")
print("\\toprule")
print("Dataset & Comparison & McNemar $p$ & DeLong $p$ \\\\")
print("\\midrule")
def fmt_p(p):
    return "$<0.0001$" if p < 0.0001 else f"${p:.4f}$"
pairs = [
    ("adult", ["XGBoost", "LightGBM", "CatBoost", "MLP", "FT-Transformer",
               "CNN-M1", "CNN-M1c", "CNN-M3-noG", "CNN-M3-noB",
               "CNN-M3-corrB", "CNN-M3-shapB", "CNN-IGTD"]),
    ("german", ["CNN-M1", "CNN-M1c", "CNN-M3-noG", "CNN-M3-noB",
                "CNN-M3-corrB", "CNN-M3-RG", "CNN-IGTD", "MLP"]),
    ("telco", ["CNN-M1", "CNN-M1c", "CNN-M3-noG", "CNN-M3-noB",
               "CNN-M3-corrB", "CNN-M3-shapB", "CNN-M3-RG", "CNN-IGTD", "MLP"]),
    ("bank", ["XGBoost", "MLP", "FT-Transformer", "CNN-M1", "CNN-M1c",
              "CNN-M3-noG", "CNN-M3-noB", "CNN-M3-shapB", "CNN-M3-RG", "CNN-IGTD"]),
    ("credit", ["CNN-M1", "CNN-M1c", "CNN-M3-noG", "CNN-M3-noB",
                "CNN-M3-corrB", "CNN-M3-shapB", "CNN-M3-RG", "CNN-IGTD", "MLP"]),
    ("sick", ["CNN-M1", "CNN-M1c", "CNN-M3-noG", "CNN-M3-noB",
              "CNN-M3-corrB", "CNN-M3-RG", "CNN-IGTD", "MLP", "FT-Transformer"]),
    ("wine", ["MLP", "FT-Transformer", "CNN-M1", "CNN-M3-noB",
              "CNN-M3-corrB", "CNN-M3-RG", "CNN-IGTD", "XGBoost"]),
]
for ds, methods in pairs:
    for m in methods:
        v = sig.get(ds, {}).get(m)
        if v is None:
            continue
        print(f"{DISPLAY[ds]} & {m} vs.\\ M3-full & {fmt_p(v['mcnemar_p'])} & {fmt_p(v['delong_p'])} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()

# Wilcoxon table
w = json.load(open(os.path.join(DATA_DIR, "extended_wilcoxon.json")))
labels = {
    ("CNN-M1", "CNN-M3-full"): "CNN-M1 vs.\\ M3-full (categorical inclusion, composite)",
    ("CNN-M3-noG", "CNN-M3-full"): "CNN-M3-noG vs.\\ M3-full (categorical channel)",
    ("CNN-M1c", "CNN-M3-RG"): "CNN-M1c vs.\\ M3-RG (channel separation)",
    ("CNN-M3-RG", "CNN-M3-full"): "CNN-M3-RG vs.\\ M3-full (B channel)",
    ("CNN-M3-noB", "CNN-M3-full"): "CNN-M3-noB vs.\\ M3-full (reordering)",
    ("CNN-M3-corrB", "CNN-M3-full"): "CNN-M3-corrB vs.\\ M3-full (linear weights)",
    ("CNN-M3-shapB", "CNN-M3-full"): "CNN-M3-shapB vs.\\ M3-full (SHAP weights)",
    ("MLP", "CNN-M3-full"): "MLP vs.\\ M3-full (image form)",
    ("FT-Transformer", "MLP"): "FT-Transformer vs.\\ MLP (strong tabular baseline)",
    ("XGBoost", "CNN-M3-full"): "XGBoost vs.\\ M3-full (trees vs.\\ image)",
}
print("\\begin{table}[t]")
print("\\footnotesize\\setlength{\\tabcolsep}{3pt}")
print("\\centering")
print("\\caption{Cross-dataset paired Wilcoxon signed-rank test on per-seed AUC pooled across seeds and datasets, with Holm--Bonferroni correction over the pre-specified family. A negative median $\\Delta$AUC means the first method is worse than the second. $n$ excludes the fully numeric Wine dataset for contrasts in which the variant is degenerate there (M1c, M3-noG).}")
print("\\label{tab:wilcoxon}")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("Comparison & $n$ & median $\\Delta$AUC & $p$ \\\\")
print("\\midrule")
for p in w["pairs"]:
    lab = labels.get((p["a"], p["b"]), f"{p['a']} vs. {p['b']}")
    sig_mark = "^{\\dagger}" if p["holm_reject"] else ""
    n_suff = "$^{\\ddagger}$" if p["n"] == 21 else ""
    print(f"{lab} & {p['n']}{n_suff} & ${p['median_delta']:+.4f}$ & ${p['p']:.4f}{sig_mark}$ \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print()
print("\\medskip")
print("\\raggedright\\footnotesize $^{\\dagger}$Significant after Holm--Bonferroni correction at $\\alpha=0.05$. $^{\\ddagger}$Rows with $n=21$ exclude the fully numeric Wine dataset, where M1c is identical to M1 and M3-noG is identical to M3-full by construction. The categorical-channel contrast (M3-noG vs.\\ M3-full) and the trees-vs-image contrast are the only ones to survive Holm correction; the categorical-inclusion result also holds at per-dataset medians ($n=7$, $p=0.031$). Channel separation, the B channel, reordering, and the image form remain non-significant even at the larger sample size, which strengthens rather than weakens the negative results. The SHAP-weight contrast is borderline at the uncorrected level ($p=0.053$) with a small negative median. Sensitivities: excluding the near-degenerate Heart pairs leaves the categorical-channel contrast significant ($p=0.0001$) and the SHAP contrast at $p=0.041$; the other negatives remain non-significant.")
print("\\end{table}")
