#!/usr/bin/env python3
"""11_resnet_analysis.py — 第二骨架（ResNet）架構魯棒性分析

載入 {name}_cnn_{tag}_resnet_results.npz 與既有 MLP，計算：
  1. 每資料集 × 編碼的 ResNet AUC（mean±std over 3 seeds）
  2. 關鍵對照的 pooled Wilcoxon（per-seed AUC，跨資料集）
  3. 關鍵對照的 per-dataset DeLong p（seed-averaged probs）
  4. 與探針結論對照（categorical / image form / channel separation）
輸出 JSON + 摘要，供稿件「Architecture robustness」小節使用。
"""
import os, json, glob
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

DATA = "data"
# 混合型（M3-noG/M1c 有定義）與數值型（M3-noG≡M3-full、M1c≡M1）
MIXED = ["adult", "bank", "credit", "german", "telco", "sick",
         "heart", "australian", "cmc", "ilpd"]
NUMERIC = ["wine", "spambase", "magic"]
KEY_TAGS = ["M1", "M1c", "M3-full", "M3-noG", "M3-RG"]


def load_auc(ds, tag, backbone="resnet", model="cnn"):
    """回傳 per-seed AUC 列表；缺檔回 None。MLP 為骨架無關對照（既有結果）。"""
    if tag == "MLP":
        f = f"{DATA}/{ds}_mlp_results.npz"
    else:
        f = f"{DATA}/{ds}_{model}_{tag}_{backbone}_results.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    y, p = z["labels"], z["per_seed_probs"]
    return [roc_auc_score(y, p[s]) for s in range(p.shape[0])]


def pooled_wilcoxon(rows):
    """rows: (ds, tag_a, tag_b) 列表 → 每個 (a,b) 的 pooled per-seed 對。
    回傳 (median_delta, p, n)。"""
    diffs = []
    for ds, a, b in rows:
        pa, pb = load_auc(ds, a), load_auc(ds, b)
        if pa is None or pb is None:
            continue
        for x, y in zip(pa, pb):
            diffs.append(x - y)
    if len(diffs) < 2 or np.all(np.array(diffs) == 0):
        return None
    res = wilcoxon(diffs)
    return float(np.median(diffs)), float(res.pvalue), len(diffs)


def delong_auc(ds, tag_a, tag_b, backbone="resnet"):
    """用 seed-averaged probs 的 placement-based DeLong 檢定（與 08 一致）。"""
    def _auc(ds, tag):
        f = f"{DATA}/{ds}_cnn_{tag}_{backbone}_results.npz"
        if not os.path.exists(f):
            return None
        z = np.load(f)
        return z["labels"], z["probs"]
    ra = _auc(ds, tag_a)
    rb = _auc(ds, tag_b)
    if ra is None or rb is None:
        return None
    la, pa = ra
    lb, pb = rb
    return delong_p(la, pa, pb)


def delong_p(y, pa, pb):
    """Placement-based DeLong 檢定（複製 08_extended_evaluate.py 的實作）。"""
    from scipy.stats import norm

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
    return float(2 * (1 - norm.cdf(abs((v10a.mean() - v10b.mean()) / np.sqrt(vd)))))


def main():
    out = {"resnet_auc": {}, "pooled": {}, "delong": {}, "probe_comparison": {}}

    # 1. ResNet AUC 表
    for ds in MIXED + NUMERIC:
        out["resnet_auc"][ds] = {}
        for tag in KEY_TAGS:
            aucs = load_auc(ds, tag)
            if aucs is not None:
                out["resnet_auc"][ds][tag] = {
                    "mean": round(float(np.mean(aucs)), 4),
                    "std": round(float(np.std(aucs)), 4),
                    "seeds": [round(a, 3) for a in aucs]}
        # MLP 對照（backbone-independent，既有結果）
        mlp = f"{DATA}/{ds}_mlp_results.npz"
        if os.path.exists(mlp):
            z = np.load(mlp); y, p = z["labels"], z["per_seed_probs"]
            out["resnet_auc"][ds]["MLP"] = {
                "mean": round(float(np.mean([roc_auc_score(y, p[s])
                                             for s in range(p.shape[0])])), 4),
                "std": round(float(np.std([roc_auc_score(y, p[s])
                                           for s in range(p.shape[0])])), 4)}

    # 2. Pooled Wilcoxon（ResNet）
    mixed_rows = [(ds, "M3-noG", "M3-full") for ds in MIXED]
    mixed_rows_noh = [(ds, "M3-noG", "M3-full") for ds in MIXED if ds != "heart"]
    cat_full = pooled_wilcoxon(mixed_rows)
    cat_noh = pooled_wilcoxon(mixed_rows_noh)
    out["pooled"]["categorical_noG_vs_full"] = {"all": cat_full, "excl_heart": cat_noh}

    out["pooled"]["imageform_mlp_vs_full"] = pooled_wilcoxon(
        [(ds, "MLP", "M3-full") for ds in MIXED + NUMERIC])
    out["pooled"]["numeric_M1_vs_full"] = pooled_wilcoxon(
        [(ds, "M1", "M3-full") for ds in MIXED + NUMERIC])
    out["pooled"]["channelsep_M1c_vs_RG"] = pooled_wilcoxon(
        [(ds, "M1c", "M3-RG") for ds in MIXED])
    out["pooled"]["channelsep_RG_vs_full"] = pooled_wilcoxon(
        [(ds, "M3-RG", "M3-full") for ds in MIXED + NUMERIC])

    # 3. DeLong（每資料集，ResNet，seed-averaged）
    for ds in MIXED + NUMERIC:
        out["delong"][ds] = {}
        for a in ["M3-noG", "MLP", "M1", "M3-RG"]:
            p = delong_auc(ds, a, "M3-full")
            if p is not None:
                out["delong"][ds][f"{a}_vs_full"] = round(p, 4)

    # 4. 與探針對照（categorical 效應量）
    probe_cat = [(ds, "M3-noG", "M3-full") for ds in MIXED]
    diffs_p = []
    for ds, a, b in probe_cat:
        pa = load_auc(ds, a, backbone="probe")
        pb = load_auc(ds, b, backbone="probe")
        if pa and pb:
            diffs_p += [x - y for x, y in zip(pa, pb)]
    diffs_r = []
    for ds, a, b in probe_cat:
        pa = load_auc(ds, a, backbone="resnet")
        pb = load_auc(ds, b, backbone="resnet")
        if pa and pb:
            diffs_r += [x - y for x, y in zip(pa, pb)]
    out["probe_comparison"]["categorical_median_delta"] = {
        "probe": round(float(np.median(diffs_p)), 4) if diffs_p else None,
        "resnet": round(float(np.median(diffs_r)), 4) if diffs_r else None}

    with open(f"{DATA}/resnet_robustness.json", "w") as fh:
        json.dump(out, fh, indent=1)

    # 摘要列印
    print("=== ResNet AUC（mean±std）===")
    hdr = "  ds".ljust(12) + "".join(f"{t:>12}" for t in ["M1", "M1c", "M3-full", "M3-noG", "M3-RG", "MLP"])
    print(hdr)
    for ds in MIXED + NUMERIC:
        row = ds.ljust(12)
        for t in KEY_TAGS + ["MLP"]:
            v = out["resnet_auc"][ds].get(t)
            if v:
                cell = "{:.3f}±{:.3f}".format(v["mean"], v["std"])
                row += f"{cell:>12}"
            else:
                row += f"{'-':>12}"
        print(row)
    print("\n=== Pooled Wilcoxon（ResNet）===")
    for k, v in out["pooled"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}[{kk}]: {'n=%d med=%.4f p=%.4f' % (vv[2], vv[0], vv[1]) if vv else 'N/A'}")
        else:
            print(f"  {k}: {'n=%d med=%.4f p=%.4f' % (v[2], v[0], v[1]) if v else 'N/A'}")


if __name__ == "__main__":
    main()
