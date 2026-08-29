"""
03_train_baseline.py
============================================================
基線模型：XGBoost 於原始表格資料上的分類（支援 adult / heart）

  流程：載入預處理資料 -> 標準化數值特徵 -> 訓練 XGBoost（多次重複、
        不同隨機種子）-> 輸出效能指標（mean ± std）與混淆矩陣圖
        -> 儲存特徵重要度（供 M3 權重重排）與各次預測機率

執行：python3 03_train_baseline.py [--dataset adult]
      python3 03_train_baseline.py --dataset heart
輸出：figures/fig_baseline_confusion_{name}.png、
      data/{name}_xgb_importance.npz、data/{name}_xgb_results.npz
============================================================
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "PingFang SC",
                                   "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from datasets import DATASETS, DATA_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 123, 2024]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]


def compute_metrics(y_true, y_pred, y_prob):
    """回傳各指標 dict。"""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_prob),
    }


def main():
    ap = argparse.ArgumentParser(description="訓練 XGBoost 基線")
    ap.add_argument("--dataset", default="adult", choices=list(DATASETS.keys()))
    args = ap.parse_args()
    name = args.dataset
    cfg = DATASETS[name]

    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_test, y_test = npz["X_test"], npz["y_test"]
    n_num = len(cfg["numeric"])
    feature_names = np.array(cfg["numeric"] + cfg["categorical"])

    # 數值特徵標準化（示範完整前處理；XGBoost 對單調變換不敏感）
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[:, :n_num] = scaler.fit_transform(X_train_s[:, :n_num])
    X_test_s[:, :n_num] = scaler.transform(X_test_s[:, :n_num])

    print(f"===== 資料集：{cfg['display']} | 基線 XGBoost =====")
    all_probs, all_metrics = [], []
    for seed in SEEDS:
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", early_stopping_rounds=20,
            random_state=seed, n_jobs=-1,
        )
        # 洩漏防護：以分層 10% 驗證集早停（原版誤用測試集早停，已修正與 07 一致）
        Xtr_s, Xval, ytr_s, yval = train_test_split(
            X_train_s, y_train, test_size=0.1, stratify=y_train,
            random_state=seed)
        model.fit(Xtr_s, ytr_s, eval_set=[(Xval, yval)], verbose=False)
        prob = model.predict_proba(X_test_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        all_probs.append(prob)
        all_metrics.append(compute_metrics(y_test, pred, prob))
        print(f"  seed={seed}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in all_metrics[-1].items()))

    all_probs = np.stack(all_probs)
    mean_probs = all_probs.mean(axis=0)
    mean_pred = (mean_probs >= 0.5).astype(int)

    # mean ± std 彙總
    print("\n  [彙總] mean ± std（n=3 種子）：")
    for i, k in enumerate(METRIC_NAMES):
        vals = np.array([m[k] for m in all_metrics])
        print(f"    {k:10s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # ---------- 混淆矩陣圖（以平均預測繪製） ----------
    cm = confusion_matrix(y_test, mean_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["負例", "正例"])
    ax.set_yticklabels(["負例", "正例"])
    ax.set_xlabel("預測標籤"); ax.set_ylabel("真實標籤")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"XGBoost 混淆矩陣（{cfg['display']}）")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_baseline_confusion_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已儲存 fig_baseline_confusion_{name}.png")

    # ---------- 特徵重要度 ----------
    # 以最後一個種子（或任一）的模型輸出重要度，作為 M3 權重重排依據
    imp = model.feature_importances_
    order = np.argsort(-imp)
    print("\n特徵重要度（XGBoost，前 8 名）：")
    for i in order[:8]:
        print(f"    {feature_names[i]:18s} {imp[i]:.4f}")

    # SHAP 平均 |貢獻| 特徵重要度（非線性、具備理論保證的模型解釋）
    print("\n計算 SHAP 特徵重要度 ...")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_train_s)
        # 二元分類可能回傳 list（類別 0/1）或單一陣列；取正類別貢獻
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        shap_imp = np.mean(np.abs(sv), axis=0)
        print("SHAP 平均 |貢獻|（前 8 名）：")
        for i in np.argsort(-shap_imp)[:8]:
            print(f"    {feature_names[i]:18s} {shap_imp[i]:.4f}")
    except Exception as e:  # 任何失敗則退回 XGBoost 重要度（不中斷流程）
        print(f"  [警告] SHAP 計算失敗（{e}），退回 XGBoost 重要度")
        shap_imp = imp

    np.savez(os.path.join(DATA_DIR, f"{name}_xgb_importance.npz"),
             importance=imp,
             importance_numeric=imp[:n_num],
             shap_importance=shap_imp,
             shap_importance_numeric=shap_imp[:n_num],
             feature_names=feature_names)
    np.savez(os.path.join(DATA_DIR, f"{name}_xgb_results.npz"),
             probs=mean_probs, labels=y_test,
             per_seed_probs=all_probs,
             per_seed_metrics=np.array([[m[k] for k in METRIC_NAMES]
                                        for m in all_metrics]),
             metric_names=np.array(METRIC_NAMES))
    print("\n已儲存 data/{}_xgb_importance.npz 與 {}_xgb_results.npz".format(name, name))


if __name__ == "__main__":
    main()
