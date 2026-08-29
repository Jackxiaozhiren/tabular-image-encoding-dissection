"""
04_train_cnn.py
============================================================
CNN 分類模型（PyTorch）— 對編碼圖像分類（支援 adult / heart + 消融變體）

  基本編碼：
    M1 灰度熱圖   -> 1 通道 64x16 CNN
    M2 4x4 拼貼   -> 1 通道 4x4  CNN（小型架構）
    M3 多通道編碼 -> 3 通道 64x16 CNN

  M3 消融變體（--variant）：
    full   依 XGBoost 重要度重排 + 類別 one-hot 通道（完整方法）
    noG    移除 G 通道（類別 one-hot 置零）           -> 驗證類別通道貢獻
    noB    不重排（B 通道 = R 通道原順序）             -> 驗證權重重排貢獻
    corrB  依線性相關係數重排（非線性 vs 線性排序）    -> 驗證非線性重要度
    shapB  依 SHAP 平均 |貢獻| 重排                    -> 更佳的非線性權重來源

  訓練協議：Adam（lr=1e-3、weight_decay=1e-4）、訓練/驗證切分（10%）、
            Early Stopping（patience 7）、可選類別加權（--balanced）。
  多次種子（預設 42,123,2024）平均預測，輸出 mean ± std。

執行：python3 04_train_cnn.py --dataset adult [--variants full]
      python3 04_train_cnn.py --dataset heart --balanced
輸出：figures/fig_cnn_confusion_{name}_{tag}.png、checkpoints/cnn_{name}_{tag}.pt、
      data/{name}_cnn_{tag}_results.npz
============================================================
"""
import argparse
import copy
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "PingFang SC",
                                   "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from datasets import DATASETS, DATA_DIR
from tabular_to_image import encode_m1, encode_m2, encode_m3, encode_m1_cat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 123, 2024]
WIDTH, HEIGHT, GRID = 64, 16, 4
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def enc_height(cfg) -> int:
    """每資料集編碼高度：至少 16 列，且能容納全部數值+類別列。

    M1c 需 n_num+n_cat 列；M3 的 R/G 通道各需 n_num/n_cat 列。
    Credit（22 數值 + 1 類別）需要 23 列，故不能用固定 16。
    """
    return max(HEIGHT, len(cfg["numeric"]) + len(cfg["categorical"]))

# 消融變體說明
VARIANTS = ["full", "noG", "noB", "corrB", "shapB", "RG"]


class TabularImageDataset(Dataset):
    """把編碼圖像 + 標籤包成 PyTorch Dataset。"""
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_cnn(in_channels, backbone="probe"):
    """CNN 骨架：小型 3 層探針（預設）或 ResNet 風格第二骨架。輸入 (C, H, W)。"""
    if backbone == "resnet":
        return build_resnet(in_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 32), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(32, 1),
    )


class _ResBlock(nn.Module):
    """殘差塊：主路徑兩層 3x3 卷積，stride 用於下採樣；shortcut 為 1x1。"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.skip = nn.Identity() if (stride == 1 and in_ch == out_ch) else \
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))


def build_resnet(in_channels):
    """ResNet 風格第二骨架：殘差塊 + 更深更寬，作為架構條件性檢查。"""
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.b1 = _ResBlock(32, 32)
            self.b2 = _ResBlock(32, 64, stride=2)
            self.b3 = _ResBlock(64, 128, stride=2)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(128, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):
            return self.head(self.b3(self.b2(self.b1(self.stem(x)))))

    return ResNet()


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_prob),
    }


def _weight_sources(name, cfg, X_train_num, y_train):
    """回傳 (importance, shap, corr) 三個權重向量（皆針對數值特徵）。"""
    imp = np.load(os.path.join(DATA_DIR, f"{name}_xgb_importance.npz"))
    w_imp = imp["importance_numeric"][:len(cfg["numeric"])]
    w_shap = imp["shap_importance_numeric"][:len(cfg["numeric"])]
    w_corr = np.array([np.corrcoef(X_train_num[:, k], y_train)[0, 1]
                       for k in range(X_train_num.shape[1])])
    return w_imp, w_shap, w_corr


def make_dataset(mode, variant, name, cfg):
    """依編碼模式與消融變體產出 (訓練圖像, 測試圖像)。"""
    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    X_train_num, X_test_num = npz["X_train_num"], npz["X_test_num"]
    X_train_cat, X_test_cat = npz["X_train_cat"], npz["X_test_cat"]
    y_train = npz["y_train"]
    w_imp, w_shap, w_corr = _weight_sources(name, cfg, X_train_num, y_train)

    def cat_one_hot(cat_matrix):
        ohs = []
        for i in range(cat_matrix.shape[0]):
            row = []
            for c in range(cat_matrix.shape[1]):
                oh = np.zeros(int(cat_matrix[:, c].max()) + 1)
                oh[int(cat_matrix[i, c])] = 1.0
                row.append(oh)
            ohs.append(row)
        return ohs

    if mode == "M1":
        # M1 高度 = 數值特徵數；墊到至少 4 列，避免低數值特徵資料集
        # （如 cmc 僅 2 個數值特徵）在兩次 MaxPool2d 後高度塌縮為 0
        m1_h = max(len(cfg["numeric"]), 4)
        def _pad_m1(v):
            img = encode_m1(v, WIDTH)[0]
            if img.shape[0] < m1_h:
                img = np.pad(img, ((0, m1_h - img.shape[0]), (0, 0)))
            return img
        X_tr = np.stack([_pad_m1(v) for v in X_train_num])[:, None]
        X_te = np.stack([_pad_m1(v) for v in X_test_num])[:, None]
    elif mode == "M1c":
        # 單通道類別包容式編碼（驗證「多通道分離」是否必要）
        oh_tr, oh_te = cat_one_hot(X_train_cat), cat_one_hot(X_test_cat)
        H = enc_height(cfg)
        X_tr = np.stack([encode_m1_cat(v, oh_tr[i], WIDTH, H)
                         for i, v in enumerate(X_train_num)])
        X_te = np.stack([encode_m1_cat(v, oh_te[i], WIDTH, H)
                         for i, v in enumerate(X_test_num)])
    elif mode == "M2":
        X_tr = np.stack([encode_m2(v, w_corr, GRID)[0] for v in X_train_num])[:, None]
        X_te = np.stack([encode_m2(v, w_corr, GRID)[0] for v in X_test_num])[:, None]
    else:  # M3 系列
        oh_tr, oh_te = cat_one_hot(X_train_cat), cat_one_hot(X_test_cat)
        # 依變體決定權重來源、G 通道與 B 通道
        zero_b = (variant == "RG")             # 純兩通道 R+G 控制（B 置零）
        if variant in ("noB", "RG"):
            w = None                           # 不重排
        elif variant == "corrB":
            w = w_corr                         # 線性排序
        elif variant == "shapB":
            w = w_shap                         # SHAP 排序
        else:                                  # full / noG
            w = w_imp                          # XGBoost 重要度
        use_cat = variant != "noG"             # noG 移除類別通道
        H = enc_height(cfg)
        X_tr = np.stack([encode_m3(v, oh_tr[i], WIDTH, H, w, use_cat, zero_b)
                         for i, v in enumerate(X_train_num)])
        X_te = np.stack([encode_m3(v, oh_te[i], WIDTH, H, w, use_cat, zero_b)
                         for i, v in enumerate(X_test_num)])
    return X_tr, X_te


def train_one_epoch(model, loader, criterion, optimizer):
    """單一 epoch：前向 -> 損失 -> 反向 -> 更新。回傳平均損失。"""
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        if len(labels) < 2:      # BatchNorm 需每通道 >1 樣本；跳過不完整 batch
            continue
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader):
    """測試/驗證評估，回傳 (平均損失, 預測機率, 標籤)。"""
    model.eval()
    probs, labels_all, total_loss, n = [], [], 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images).squeeze(1)
        total_loss += nn.BCEWithLogitsLoss()(logits, labels).item() * len(labels)
        n += len(labels)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    return total_loss / n, np.concatenate(probs), np.concatenate(labels_all)


def run_cnn(mode, variant, name, cfg, epochs=30, batch_size=256,
            balanced=False, patience=7, verbose=True, backbone="probe"):
    """多種子訓練並評估單一編碼/變體，回傳 (指標 mean±std, 平均機率, 標籤)。"""
    tag = f"{mode}" if mode != "M3" else f"M3-{variant}"
    out_tag = tag if backbone == "probe" else f"{tag}_{backbone}"
    print(f"\n===== 訓練 CNN（{tag}/{backbone}，{cfg['display']}） =====")
    X_tr, X_te = make_dataset(mode, variant, name, cfg)
    in_ch = 3 if mode == "M3" else 1
    test_ds = TabularImageDataset(X_te, np.load(
        os.path.join(DATA_DIR, f"{name}_arrays.npz"))["y_test"])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_probs, all_metrics = [], []
    for seed in SEEDS:
        # 訓練/驗證切分（分層、與種子綁定，可重現）
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            X_tr, np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))["y_train"],
            test_size=0.1, stratify=np.load(
                os.path.join(DATA_DIR, f"{name}_arrays.npz"))["y_train"],
            random_state=seed)
        train_loader = DataLoader(TabularImageDataset(X_tr_sub, y_tr_sub),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TabularImageDataset(X_val, y_val),
                                batch_size=batch_size, shuffle=False)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_cnn(in_ch, backbone).to(DEVICE)
        pos_weight = None
        if balanced:
            n_pos, n_neg = y_tr_sub.sum(), (y_tr_sub == 0).sum()
            # float32 explicitly: MPS does not support float64 tensors
            pos_weight = torch.tensor([float(n_neg) / float(max(n_pos, 1))],
                                      dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                     weight_decay=1e-4)

        # Early Stopping：以驗證損失監看最佳模型
        best_val, best_state, no_improve = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, _, _ = evaluate(model, val_loader)
            if val_loss < best_val - 1e-4:
                best_val, best_state, no_improve = val_loss, \
                    copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  seed={seed} epoch {epoch:2d}  早停 "
                              f"(訓練損失 {loss:.4f}, 驗證損失 {val_loss:.4f})")
                    break
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  seed={seed} epoch {epoch:2d}/{epochs}  "
                      f"訓練損失 = {loss:.4f} 驗證損失 = {val_loss:.4f}")
        if best_state is not None:
            model.load_state_dict(best_state)

        _, prob, labels = evaluate(model, test_loader)
        pred = (prob >= 0.5).astype(int)
        all_probs.append(prob)
        all_metrics.append(compute_metrics(labels, pred, prob))
        if verbose:
            print(f"  seed={seed} 測試: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in all_metrics[-1].items()))

    all_probs = np.stack(all_probs)
    mean_probs = all_probs.mean(axis=0)
    mean_metrics = compute_metrics(labels, (mean_probs >= 0.5).astype(int),
                                   mean_probs)
    std_str = "、".join(
        f"{k}={np.mean([m[k] for m in all_metrics]):.4f}±"
        f"{np.std([m[k] for m in all_metrics]):.4f}" for k in METRIC_NAMES)
    print(f"  [彙總 mean ± std] {std_str}")

    # ---------- 混淆矩陣圖（平均預測） ----------
    cm = confusion_matrix(labels, (mean_probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["負例", "正例"]); ax.set_yticklabels(["負例", "正例"])
    ax.set_xlabel("預測標籤"); ax.set_ylabel("真實標籤")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"CNN（{tag}）混淆矩陣（{cfg['display']}）")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_cnn_confusion_{name}_{out_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    torch.save(model.state_dict(),
               os.path.join(CKPT_DIR, f"cnn_{name}_{out_tag}.pt"))
    np.savez(os.path.join(DATA_DIR, f"{name}_cnn_{out_tag}_results.npz"),
             probs=mean_probs, labels=labels, per_seed_probs=all_probs,
             per_seed_metrics=np.array([[m[k] for k in METRIC_NAMES]
                                        for m in all_metrics]),
             metric_names=np.array(METRIC_NAMES))
    print(f"  已儲存 fig_cnn_confusion_{name}_{out_tag}.png、"
          f"cnn_{name}_{out_tag}.pt、{name}_cnn_{out_tag}_results.npz")
    return mean_metrics


def main():
    ap = argparse.ArgumentParser(description="對編碼圖像訓練 CNN（含消融）")
    ap.add_argument("--dataset", default="adult", choices=list(DATASETS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--balanced", action="store_true",
                    help="使用類別加權損失（小樣本資料集建議）")
    ap.add_argument("--variants", default="full",
                    help="逗號分隔的 M3 消融變體：full,noG,noB,corrB,shapB")
    ap.add_argument("--tags", default=None,
                    help="（可選）直接指定要訓練的 tag 清單（如 M1,M2,M3-full,M3-noG），"
                         "指定後忽略 --variants；方便中斷後續跑")
    ap.add_argument("--backbone", default="probe", choices=["probe", "resnet"],
                    help="CNN 骨架：probe（小型 3 層，預設）或 resnet（ResNet 風格第二骨架）")
    args = ap.parse_args()
    name, cfg = args.dataset, DATASETS[args.dataset]

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = ["M1", "M2"]
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        tags += [f"M3-{v}" for v in variants if v in VARIANTS]

    results = {}
    failures = []
    bb = args.backbone
    for tag in tags:
        try:
            if tag == "M1":
                results[tag] = run_cnn("M1", None, name, cfg, args.epochs,
                                       balanced=args.balanced, backbone=bb)
            elif tag == "M1c":
                results[tag] = run_cnn("M1c", None, name, cfg, args.epochs,
                                       balanced=args.balanced, backbone=bb)
            elif tag == "M2":
                results[tag] = run_cnn("M2", None, name, cfg, args.epochs,
                                       balanced=args.balanced, backbone=bb)
            elif tag.startswith("M3-"):
                v = tag[3:]
                if v not in VARIANTS:
                    print(f"[警告] 忽略未知變體: {v}")
                    continue
                results[tag] = run_cnn("M3", v, name, cfg, args.epochs,
                                       balanced=args.balanced, backbone=bb)
            else:
                print(f"[警告] 忽略未知 tag: {tag}")
        except Exception as e:
            failures.append(tag)
            print(f"[錯誤] tag={tag} 訓練失敗：{e}")

    print("\n===== 彙總（mean 指標） =====")
    for tag, m in results.items():
        print(f"  {tag:8s}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    if failures:
        print(f"[致命] 以下 tag 訓練失敗、結果檔不完整：{failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
