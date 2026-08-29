"""
07_train_modern_baselines.py
============================================================
阶段三实验强化：现代基线补强（Neurocomputing 审稿三大支柱①）

  1. GBDT 家族补强：LightGBM、CatBoost（与 XGBoost 同协议，3 种子）
  2. 决定性对照：MLP——吃「MinMax 数值 + One-Hot 类别」的同一特征矩阵，
     与影像编码（M3 的 R+G 通道）完全同特征，检验「影像形式是否必要」
  3. XGBoost 以「验证集早停」重跑（原 03 用测试集早停，属协议瑕疵，
     此处统一为分层 10% 验证集早停，消除数据泄漏）

输出（沿用 05 的 npz 格式）：
  data/{name}_xgb_results.npz   （重跑，val-ES）
  data/{name}_lgb_results.npz
  data/{name}_cat_results.npz
  data/{name}_mlp_results.npz   （one-hot + MinMax 特征）
============================================================
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from datasets import DATASETS, DATA_DIR
from ft_transformer import train_ft

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 2024]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def load_data(name):
    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    return (npz["X_train"], npz["y_train"], npz["X_test"], npz["y_test"],
            npz["X_train_num"], npz["X_test_num"],
            npz["X_train_cat"], npz["X_test_cat"])


def one_hot_cats(X_cat, cardinalities=None):
    """把 label-encoded 类别矩阵转成 one-hot。回传稀疏友好的 (n, sum(n_cat)) float32。
    默认以该矩阵的最大索引 +1 决定维度；传入 cardinalities（须以训练集计算）
    则 train/test 共用同一维度，避免 train/test 基数不一致导致宽度错位。"""
    parts = []
    for j in range(X_cat.shape[1]):
        col = X_cat[:, j]
        n_cat = cardinalities[j] if cardinalities is not None else int(np.max(col)) + 1
        oh = np.zeros((len(col), n_cat), dtype=np.float32)
        oh[np.arange(len(col)), col.astype(int)] = 1.0
        parts.append(oh)
    return np.hstack(parts) if parts else np.zeros((len(X_cat), 0), dtype=np.float32)


def metrics(y_true, y_pred, y_prob):
    return [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred),
            recall_score(y_true, y_pred), f1_score(y_true, y_pred),
            roc_auc_score(y_true, y_prob)]


def save_results(name, tag, per_seed_probs, y_test):
    mean_prob = np.mean(np.stack(per_seed_probs), axis=0)
    per_seed_metrics = np.array(
        [metrics(y_test, (p >= 0.5).astype(int), p) for p in per_seed_probs])
    np.savez_compressed(
        os.path.join(DATA_DIR, f"{name}_{tag}_results.npz"),
        probs=mean_prob.astype(np.float32), labels=y_test,
        per_seed_probs=np.stack(per_seed_probs).astype(np.float32),
        per_seed_metrics=per_seed_metrics, metric_names=np.array(METRIC_NAMES))
    print(f"  [saved] {name}_{tag}_results.npz  "
          f"mean-AUC={roc_auc_score(y_test, mean_prob):.4f} | per-seed="
          + ",".join(f"{roc_auc_score(y_test, p):.4f}" for p in per_seed_probs))


def train_trees(name, X_train, y_train, X_test, y_test, n_num):
    """LightGBM / CatBoost（含 XGBoost 重跑）：StandardScaler 数值 + label-encoded 类别。
    统一「分层 10% 验证集早停」协议（消除原 03 用测试集早停的泄漏）。"""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    scaler = StandardScaler()
    Xtr = X_train.copy().astype(np.float64)
    Xte = X_test.copy().astype(np.float64)
    Xtr[:, :n_num] = scaler.fit_transform(Xtr[:, :n_num])
    Xte[:, :n_num] = scaler.transform(Xte[:, :n_num])

    specs = {
        "xgb": (XGBClassifier, dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="logloss")),
        "lgb": (LGBMClassifier, dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     verbose=-1)),
        "cat": (CatBoostClassifier, dict(iterations=300, depth=4, learning_rate=0.1,
                                         verbose=0, allow_writing_files=False)),
    }
    for tag, (Cls, kw) in specs.items():
        per_seed = []
        for seed in SEEDS:
            Xtr_s, Xval, ytr_s, yval = train_test_split(
                Xtr, y_train, test_size=0.1, stratify=y_train, random_state=seed)
            if tag == "cat":
                m = Cls(**kw, random_seed=seed, early_stopping_rounds=20)
            else:
                m = Cls(**kw, random_state=seed, early_stopping_rounds=20,
                        n_jobs=-1)
            m.fit(Xtr_s, ytr_s, eval_set=[(Xval, yval)])
            per_seed.append(m.predict_proba(Xte)[:, 1])
        save_results(name, tag, per_seed, y_test)


def train_mlp(name, X_train_num, X_train_cat, y_train,
              X_test_num, X_test_cat, y_test):
    """MLP 决定性对照：输入 = MinMax 数值 + One-Hot 类别（与 M3 的 R+G 通道同特征）。
    Adam lr=1e-3、wd=1e-4、batch 256、分层 10% 验证集早停（patience 7，max 40 epochs），
    与 CNN 协议一致。"""
    card = [int(X_train_cat[:, j].max()) + 1 for j in range(X_train_cat.shape[1])]
    Xtr = np.hstack([X_train_num, one_hot_cats(X_train_cat, card)])
    Xte = np.hstack([X_test_num, one_hot_cats(X_test_cat, card)])
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    yte_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    def build_mlp():
        d = Xtr.shape[1]
        return nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        # 分层验证集（与 CNN 协议一致：random_state=seed）
        idx = np.arange(len(y_train))
        tr_idx, va_idx = train_test_split(idx, test_size=0.1,
                                          stratify=y_train, random_state=seed)
        model = build_mlp().to(DEVICE)
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xtr_t_, Xva_t = Xtr_t[tr_idx], Xtr_t[va_idx]
        ytr_t_, yva_t = ytr_t[tr_idx], ytr_t[va_idx]
        best_val, best_state, no_imp = float("inf"), None, 0
        for epoch in range(40):
            model.train()
            perm = torch.randperm(len(tr_idx))
            for b in torch.split(perm, 256):
                if len(b) < 2:      # BatchNorm 需要每通道 >1 樣本
                    continue
                opt.zero_grad()
                loss = crit(model(Xtr_t_[b]).squeeze(-1), ytr_t_[b])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vloss = crit(model(Xva_t).squeeze(-1), yva_t).item()
            if vloss < best_val - 1e-4:
                best_val, best_state, no_imp = vloss, {k: v.clone()
                                                       for k, v in model.state_dict().items()}, 0
            else:
                no_imp += 1
                if no_imp >= 7:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            per_seed.append(torch.sigmoid(model(Xte_t).squeeze(-1)).cpu().numpy())
    save_results(name, "mlp", per_seed, y_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="adult,heart,wine")
    ap.add_argument("--skip-trees", action="store_true",
                    help="跳过 GBDT 家族（只跑 MLP/FT）")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-ft", action="store_true", help="跳过 FT-Transformer")
    args = ap.parse_args()

    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        print(f"===== {DATASETS[name]['display']} =====")
        X_train, y_train, X_test, y_test, Xn_tr, Xn_te, Xc_tr, Xc_te = load_data(name)
        n_num = len(DATASETS[name]["numeric"])
        if not args.skip_trees:
            train_trees(name, X_train, y_train, X_test, y_test, n_num)
        if not args.skip_mlp:
            train_mlp(name, Xn_tr, Xc_tr, y_train, Xn_te, Xc_te, y_test)
        if not args.skip_ft:
            train_ft(name, Xn_tr, Xc_tr, y_train, Xn_te, Xc_te, y_test)


if __name__ == "__main__":
    main()
