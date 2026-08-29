"""
10_published_encodings.py
============================================================
Train the shared CNN probe on the PUBLISHED IGTD encoding (Zhu et al. 2021,
Scientific Reports) as a sanity check on the design-space dissection.

The paper's M1/M2 are hand-crafted numeric-only encodings; IGTD is the
published numeric-only encoder that M2 approximates ("simplified IGTD-style
layout"). Running the real IGTD (via the clean port in igtd_encoder.py)
shows where a published numeric-only encoder lands relative to the
categorical-inclusive encodings.

Protocol: identical to script 04 (10% stratified validation split, early
stopping patience 7, Adam lr=1e-3 wd=1e-4, batch 256, 3 seeds, class-weighted
loss for Heart). Output: data/{name}_cnn_IGTD_results.npz + the optimized
layout in data/{name}_igtd_layout.npz.

Usage:  python3 10_published_encodings.py [--datasets adult,heart,wine,bank,credit]
"""
import argparse
import copy
import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import igtd_encoder
from datasets import DATASETS, DATA_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_t04():
    spec = importlib.util.spec_from_file_location("t04", os.path.join(BASE_DIR, "04_train_cnn.py"))
    t04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t04)
    return t04


t04 = _load_t04()
SEEDS = t04.SEEDS
METRIC_NAMES = t04.METRIC_NAMES
DEVICE = t04.DEVICE
GRID = 8


class _ImgDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def run_igtd(name, cfg, epochs=30, batch_size=256, balanced=False,
             patience=7, verbose=True):
    npz = np.load(os.path.join(DATA_DIR, f"{name}_arrays.npz"))
    Xtr, Xte = npz["X_train_num"], npz["X_test_num"]
    ytr, yte = npz["y_train"], npz["y_test"]

    # IGTD layout from the (min-max scaled) TRAINING features only.
    arrangement, coords = igtd_encoder.igtd_layout(Xtr, num_r=GRID, num_c=GRID)
    itr = igtd_encoder.encode_igtd(Xtr, arrangement, coords, GRID, GRID)
    ite = igtd_encoder.encode_igtd(Xte, arrangement, coords, GRID, GRID)
    np.savez(os.path.join(DATA_DIR, f"{name}_igtd_layout.npz"),
             arrangement=arrangement, rows=coords[0], cols=coords[1], grid=GRID)

    if verbose:
        print(f"===== CNN-IGTD ({cfg['display']}) =====")
    all_probs, all_metrics = [], []
    for seed in SEEDS:
        Xtr_s, Xval, ytr_s, yval = train_test_split(
            itr, ytr, test_size=0.1, stratify=ytr, random_state=seed)
        tr_loader = DataLoader(_ImgDataset(Xtr_s, ytr_s), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(_ImgDataset(Xval, yval), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(_ImgDataset(ite, yte), batch_size=batch_size, shuffle=False)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = t04.build_cnn(1).to(DEVICE)
        if balanced:
            n_pos, n_neg = ytr_s.sum(), (ytr_s == 0).sum()
            # float32 explicitly: MPS does not support float64 tensors
            pos_weight = torch.tensor([float(n_neg) / float(max(n_pos, 1))],
                                      dtype=torch.float32).to(DEVICE)
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val, best_state, no_improve = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for images, labels in tr_loader:
                if len(labels) < 2:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images).squeeze(1), labels)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                vprobs, vlabels, vloss = [], [], 0.0
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    logits = model(images).squeeze(1)
                    vloss += criterion(logits, labels).item() * len(labels)
                    vprobs.append(torch.sigmoid(logits).cpu().numpy())
                    vlabels.append(labels.cpu().numpy())
                vloss /= max(len(vlabels[0]) if vlabels else 1, 1)
            if vloss < best_val - 1e-4:
                best_val, best_state, no_improve = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            test_probs = []
            for images, _ in test_loader:
                test_probs.append(torch.sigmoid(model(images.to(DEVICE)).squeeze(1)).cpu().numpy())
        prob = np.concatenate(test_probs)
        pred = (prob >= 0.5).astype(int)
        all_probs.append(prob)
        all_metrics.append(t04.compute_metrics(yte, pred, prob))

    all_probs = np.stack(all_probs)
    mean_probs = all_probs.mean(axis=0)
    mean_pred = (mean_probs >= 0.5).astype(int)
    mean_metrics = t04.compute_metrics(yte, mean_pred, mean_probs)
    if verbose:
        print("  [mean ± std] " + "、".join(
            f"{k}={np.mean([m[k] for m in all_metrics]):.4f}±"
            f"{np.std([m[k] for m in all_metrics]):.4f}" for k in METRIC_NAMES))

    np.savez(os.path.join(DATA_DIR, f"{name}_cnn_IGTD_results.npz"),
             probs=mean_probs, labels=yte, per_seed_probs=all_probs,
             per_seed_metrics=np.array([[m[k] for k in METRIC_NAMES]
                                        for m in all_metrics]),
             metric_names=np.array(METRIC_NAMES))
    return mean_metrics


def main():
    ap = argparse.ArgumentParser(description="Train CNN on published IGTD encoding")
    ap.add_argument("--datasets", default="adult,heart,wine,bank,credit")
    args = ap.parse_args()
    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        cfg = DATASETS[name]
        run_igtd(name, cfg, balanced=(name == "heart"))


if __name__ == "__main__":
    main()
