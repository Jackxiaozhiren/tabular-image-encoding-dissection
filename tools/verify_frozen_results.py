"""Lightweight integrity checks for manuscript-active aggregate evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "frozen_results"


def auc(dataset: str, method: str) -> float:
    with (RESULTS / "auc_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["method"] == method:
                return float(row["auc_mean"])
    raise KeyError((dataset, method))


def main() -> None:
    assert abs(auc("adult", "CNN-M3-full") - 0.9091) < 5e-4
    assert abs(auc("adult", "CNN-M3-noG") - 0.8471) < 5e-4
    assert abs(auc("bank", "CNN-M3-full") - 0.9249) < 5e-4
    assert abs(auc("bank", "CNN-M3-noG") - 0.8665) < 5e-4

    wilcoxon = json.loads((RESULTS / "extended_wilcoxon.json").read_text(encoding="utf-8"))
    pairs = {(p["a"], p["b"]): p for p in wilcoxon["pairs"]}
    cat = pairs[("CNN-M3-noG", "CNN-M3-full")]
    image = pairs[("MLP", "CNN-M3-full")]
    assert cat["holm_reject"] is True
    assert abs(cat["p"] - 3.801573526884496e-05) < 1e-12
    assert abs(cat["p_dataset_median"] - 0.00390625) < 1e-12
    assert image["holm_reject"] is False
    assert abs(image["p"] - 0.12559677728090787) < 1e-12

    resnet = json.loads((RESULTS / "resnet_robustness.json").read_text(encoding="utf-8"))
    # This file records the pooled ResNet analysis. The manuscript's dataset-median
    # p=0.064453125 was independently recomputed from the per-seed frozen NPZ files.
    assert resnet["pooled"]["categorical_noG_vs_full"]["all"][2] == 30
    print("Frozen DMKD aggregate evidence checks passed")


if __name__ == "__main__":
    main()
