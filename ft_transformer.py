"""
ft_transformer.py
============================================================
FT-Transformer（Gorishniy et al. 2021, arXiv:2106.11959）乾淨實作，
作為表格深度學習的強力基線（Neurocomputing house style 支柱①）。

架構：
  Feature Tokenizer（數值特徵線性投影 + 類別特徵 Embedding）+
  [CLS] token + 標準 Transformer Encoder（pre-LN、GELU）+
  MLP 頭（取 CLS token 輸出）。

訓練協定與 MLP 對照一致：3 種子、分層 10% 驗證集早停（patience 7）、
AdamW（lr=1e-4, weight_decay=1e-5）、batch 256、max 40 epochs。

輸出 data/{name}_ft_results.npz（與其他基線同格式）。
============================================================
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from datasets import DATA_DIR

SEEDS = [42, 123, 2024]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


class FTTransformer(nn.Module):
    """FT-Transformer（lite）：Feature Tokenizer + Transformer Encoder + CLS head。"""

    def __init__(self, n_num: int, cat_cardinalities, d_token: int = 192,
                 n_blocks: int = 3, n_heads: int = 8, d_ffn: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.num_linear = nn.Linear(n_num, d_token)
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cat_cardinalities])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, num, cat):
        tokens = [self.cls.expand(num.shape[0], -1, -1)]
        tokens.append(self.num_linear(num).unsqueeze(1))
        for j, emb in enumerate(self.cat_embeds):
            tokens.append(emb(cat[:, j].long()).unsqueeze(1))
        x = torch.cat(tokens, dim=1)          # (B, 1+n_num+n_cat, d)
        x = self.encoder(x)
        return self.head(x[:, 0])             # CLS token -> (B, 1)


def cat_cardinalities(X_cat: np.ndarray) -> list:
    """每類別特徵的基數 = 最大 label code + 1（全部 >0）。"""
    if X_cat.shape[1] == 0:
        return []
    return [int(X_cat[:, j].max()) + 1 for j in range(X_cat.shape[1])]


def metrics(y_true, y_pred, y_prob):
    return [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred),
            recall_score(y_true, y_pred), f1_score(y_true, y_pred),
            roc_auc_score(y_true, y_prob)]


def train_ft(name, X_train_num, X_train_cat, y_train,
             X_test_num, X_test_cat, y_test, d_token=192, n_blocks=3,
             n_heads=8, dropout=0.1, max_epochs=40, patience=7, batch_size=256,
             lr=1e-4, weight_decay=1e-5):
    """FT-Transformer 基線：與 MLP 對照相同的 3 種子 / 早停協定。"""
    cardinalities = cat_cardinalities(X_train_cat)
    Xtr_n = torch.tensor(X_train_num, dtype=torch.float32, device=DEVICE)
    Xtr_c = torch.tensor(X_train_cat, dtype=torch.float32, device=DEVICE)
    Xte_n = torch.tensor(X_test_num, dtype=torch.float32, device=DEVICE)
    Xte_c = torch.tensor(X_test_cat, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    yte_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        idx = np.arange(len(y_train))
        tr_idx, va_idx = train_test_split(idx, test_size=0.1,
                                          stratify=y_train, random_state=seed)
        model = FTTransformer(Xtr_n.shape[1], cardinalities, d_token=d_token,
                              n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        model.to(DEVICE)
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        Xtr_n_, Xva_n = Xtr_n[tr_idx], Xtr_n[va_idx]
        Xtr_c_, Xva_c = Xtr_c[tr_idx], Xtr_c[va_idx]
        ytr_t_, yva_t = ytr_t[tr_idx], ytr_t[va_idx]
        best_val, best_state, no_imp = float("inf"), None, 0
        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(len(tr_idx))
            for b in torch.split(perm, batch_size):
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = crit(model(Xtr_n_[b], Xtr_c_[b]).squeeze(-1), ytr_t_[b])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vloss = crit(model(Xva_n, Xva_c).squeeze(-1), yva_t).item()
            if vloss < best_val - 1e-4:
                best_val, best_state, no_imp = vloss, {
                    k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            per_seed.append(torch.sigmoid(
                model(Xte_n, Xte_c).squeeze(-1)).cpu().numpy())

    mean_prob = np.mean(np.stack(per_seed), axis=0)
    per_seed_metrics = np.array(
        [metrics(y_test, (p >= 0.5).astype(int), p) for p in per_seed])
    np.savez_compressed(
        os.path.join(DATA_DIR, f"{name}_ft_results.npz"),
        probs=mean_prob.astype(np.float32), labels=y_test,
        per_seed_probs=np.stack(per_seed).astype(np.float32),
        per_seed_metrics=per_seed_metrics, metric_names=np.array(METRIC_NAMES))
    print(f"  [saved] {name}_ft_results.npz  "
          f"mean-AUC={roc_auc_score(y_test, mean_prob):.4f} | per-seed="
          + ",".join(f"{roc_auc_score(y_test, p):.4f}" for p in per_seed))
