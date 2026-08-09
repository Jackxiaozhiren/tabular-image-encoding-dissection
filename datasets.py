"""
datasets.py
============================================================
資料集設定與通用前處理（供 01–05 腳本共用）

支援資料集：
  - adult : UCI Adult（Census Income），二元分類，年收入是否 >50K
  - heart : UCI Heart Disease（Cleveland），二元分類，是否罹患心臟病

每個資料集設定包含：下載網址、欄位名稱、數值/類別特徵清單、標籤欄位與
二值化規則。`prepare_dataset()` 完成「下載 -> 清洗 -> 特徵工程 -> 儲存 npz」
的完整流程，確保結果可重現。

執行：python3 01_download_clean.py --datasets adult,heart
============================================================
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"

DATASETS = {
    "adult": {
        "display": "UCI Adult (Census Income)",
        "urls": {
            "train": f"{UCI_BASE}/adult/adult.data",
            "test": f"{UCI_BASE}/adult/adult.test",
        },
        "skiprows": {"test": 1},   # adult.test 首行為標頭說明，需跳過
        "column_names": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income",
        ],
        "numeric": ["age", "fnlwgt", "education_num", "capital_gain",
                    "capital_loss", "hours_per_week"],
        "categorical": ["workclass", "education", "marital_status", "occupation",
                        "relationship", "race", "sex", "native_country"],
        "label": "income",
        "label_map": {">50K": 1},      # 其餘類別映射為 0（<=50K）
        "has_test": True,               # 官方已提供測試集
    },
    "heart": {
        "display": "UCI Heart Disease (Cleveland)",
        "urls": {"train": f"{UCI_BASE}/heart-disease/processed.cleveland.data"},
        "skiprows": {},
        "column_names": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
        ],
        "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope",
                        "ca", "thal"],
        "label": "num",
        "label_ge": 1,                  # num >= 1 視為正例（患病）
        "has_test": False,              # 需自行切分
        "test_ratio": 0.25,
    },
    "wine": {
        "display": "UCI Wine Quality (White)",
        "urls": {"train": f"{UCI_BASE}/wine-quality/winequality-white.csv"},
        "skiprows": {},
        "delimiter": ";",               # 分號分隔 + 表頭
        "has_header": True,
        "column_names": ["fixed_acidity", "volatile_acidity", "citric_acid",
                         "residual_sugar", "chlorides", "free_sulfur_dioxide",
                         "total_sulfur_dioxide", "density", "pH", "sulphates",
                         "alcohol", "quality"],
        "numeric": ["fixed_acidity", "volatile_acidity", "citric_acid",
                    "residual_sugar", "chlorides", "free_sulfur_dioxide",
                    "total_sulfur_dioxide", "density", "pH", "sulphates",
                    "alcohol"],
        "categorical": [],              # 全數值資料集（隔離檢驗權重重排）
        "label": "quality",
        "label_ge": 7,                  # quality >= 7 視為佳釀（正例）
        "has_test": False,              # 需自行切分
        "test_ratio": 0.25,
    },
    "bank": {
        "display": "UCI Bank Marketing",
        "urls": {"train": f"{UCI_BASE}/00222/bank.zip"},
        "skiprows": {},
        "delimiter": ";",
        "has_header": True,
        "column_names": ["age", "job", "marital", "education", "default",
                         "balance", "housing", "loan", "contact", "day",
                         "month", "duration", "campaign", "pdays", "previous",
                         "poutcome", "y"],
        "numeric": ["age", "balance", "day", "duration", "campaign",
                    "pdays", "previous"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "contact", "month", "poutcome"],
        "label": "y",
        "label_map": {"yes": 1},        # 其餘 -> no (0)
        "has_test": False,
        "test_ratio": 0.25,
    },
    "credit": {
        "display": "UCI Credit Card Default",
        "urls": {"train": f"{UCI_BASE}/00350/default%20of%20credit%20card%20clients.xls"},
        "skiprows": {"train": 1},       # xls 兩行表頭，跳過通用名行
        "has_header": True,
        "column_names": ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE",
                         "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                         "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                         "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
                         "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5",
                         "PAY_AMT6", "default"],
        "numeric": ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", "PAY_0",
                    "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
                    "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
                    "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
        "categorical": ["SEX"],
        "label": "default",
        "drop_columns": ["ID"],         # 無意義識別碼
        "has_test": False,
        "test_ratio": 0.25,
    },
}


def load_raw(name: str):
    """
    下載資料集原始資料（有快取則讀取快取）。
    adult 回傳 (train_df, test_df)；heart 回傳 (df, None)。
    """
def _read_csv(path_or_url, cfg, skiprows=0):
    """依資料集設定讀取資料（支援 CSV / zip 內 CSV / Excel）。"""
    url = str(path_or_url).lower()
    if url.endswith(".zip"):
        import io
        import urllib.request
        import zipfile
        with urllib.request.urlopen(path_or_url) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            return pd.read_csv(io.BytesIO(zf.read(csv_name)),
                               header=0 if cfg.get("has_header") else None,
                               delimiter=cfg.get("delimiter", ","),
                               skiprows=skiprows)
    if url.endswith((".xls", ".xlsx")):
        return pd.read_excel(path_or_url, header=0,
                             skiprows=skiprows)
    return pd.read_csv(path_or_url, header=0 if cfg.get("has_header") else None,
                       delimiter=cfg.get("delimiter", ","),
                       skiprows=skiprows)


def load_raw(name: str):
    """
    下載資料集原始資料（有快取則讀取快取）。
    adult 回傳 (train_df, test_df)；heart/wine 回傳 (df, None)。
    """
    cfg = DATASETS[name]
    cache_files = []
    if cfg["has_test"]:
        cache_files = [f"raw_{name}_train.csv", f"raw_{name}_test.csv"]
    else:
        cache_files = [f"raw_{name}.csv"]

    cached = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in cache_files)
    if not cached:
        print(f"  下載 {cfg['display']} ...")
        if cfg["has_test"]:
            train = _read_csv(cfg["urls"]["train"], cfg)
            test = _read_csv(cfg["urls"]["test"], cfg,
                             skiprows=cfg["skiprows"].get("test", 0))
            train.to_csv(os.path.join(DATA_DIR, cache_files[0]),
                         index=False, header=False)
            test.to_csv(os.path.join(DATA_DIR, cache_files[1]),
                        index=False, header=False)
            return train, test
        else:
            df = _read_csv(cfg["urls"]["train"], cfg,
                           skiprows=cfg["skiprows"].get("train", 0))
            df.to_csv(os.path.join(DATA_DIR, cache_files[0]),
                      index=False, header=False)
            return df, None
    else:
        if cfg["has_test"]:
            train = pd.read_csv(os.path.join(DATA_DIR, cache_files[0]), header=None)
            test = pd.read_csv(os.path.join(DATA_DIR, cache_files[1]), header=None)
            return train, test
        else:
            df = pd.read_csv(os.path.join(DATA_DIR, cache_files[0]), header=None)
            return df, None


def _clean(df: pd.DataFrame, columns, cfg: dict = None) -> pd.DataFrame:
    """通用清洗：欄位命名、去空白與尾端句點、缺失值標準化、去重複、移除無意義欄位。"""
    df = df.copy()
    df.columns = columns
    if cfg:
        df = _drop_columns(df, cfg)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.rstrip(".")
    df = df.replace("?", np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _binarize_label(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """標籤二值化：adult/bank 依 label_map；heart/wine 依 label_ge 門檻；
    credit 已為 0/1 直接取整。"""
    if "label_map" in cfg:
        keys = list(cfg["label_map"].keys())
        return df[cfg["label"]].astype(str).isin(keys).astype(int).values
    elif "label_ge" in cfg:
        return (df[cfg["label"]].astype(float) >= cfg["label_ge"]).astype(int).values
    else:
        return df[cfg["label"]].astype(int).values


def _drop_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """移除無意義欄位（如 Credit 的 ID）。"""
    if cfg.get("drop_columns"):
        return df.drop(columns=[c for c in cfg["drop_columns"] if c in df.columns])
    return df


def _encode_cat(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """類別特徵：以眾數填充缺失 + Label Encoding。回傳 (n, n_cat) 矩陣。"""
    X = df[cfg["categorical"]].copy()
    out = np.zeros((len(X), len(cfg["categorical"])), dtype=float)
    for j, col in enumerate(cfg["categorical"]):
        mode_val = X[col].mode()[0]
        X[col] = X[col].fillna(mode_val).astype(str)
        enc = LabelEncoder()
        enc.fit(X[col])
        out[:, j] = enc.transform(X[col])
    return out


def prepare_dataset(name: str, seed: int = 42) -> dict:
    """
    完整前處理流程，儲存 data/{name}_arrays.npz，並回傳摘要 dict。
    """
    cfg = DATASETS[name]
    print(f"===== 資料集：{cfg['display']} =====")

    train, test = load_raw(name)

    if cfg["has_test"]:
        # -------- adult：官方切分 --------
        train = _clean(train, cfg["column_names"], cfg)
        test = _clean(test, cfg["column_names"], cfg)
        y_train = _binarize_label(train, cfg)
        y_test = _binarize_label(test, cfg)

        scaler = MinMaxScaler()
        X_train_num = scaler.fit_transform(train[cfg["numeric"]].astype(float))
        X_test_num = scaler.transform(test[cfg["numeric"]].astype(float))

        # 類別編碼：以「訓練+測試」併集擬合，避免測試集新類別報錯
        cat_enc_train = _encode_cat(pd.concat([train, test], ignore_index=True), cfg)
        cat_enc_test = cat_enc_train[len(train):]
        cat_enc_train = cat_enc_train[:len(train)]
        X_train_cat, X_test_cat = cat_enc_train, cat_enc_test

        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
    else:
        # -------- heart：自行切分（分層抽樣） --------
        df = _clean(train, cfg["column_names"], cfg)
        y = _binarize_label(df, cfg)
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(df[cfg["numeric"]].astype(float))
        X_cat = _encode_cat(df, cfg)
        X = np.hstack([X_num, X_cat])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.get("test_ratio", 0.25),
            stratify=y, random_state=seed)
        X_train_num = X_train[:, :len(cfg["numeric"])]
        X_test_num = X_test[:, :len(cfg["numeric"])]
        X_train_cat = X_train[:, len(cfg["numeric"]):]
        X_test_cat = X_test[:, len(cfg["numeric"]):]

    out_path = os.path.join(DATA_DIR, f"{name}_arrays.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        X_train_num=X_train_num, X_test_num=X_test_num,
        X_train_cat=X_train_cat, X_test_cat=X_test_cat,
        train_num_min=scaler.data_min_, train_num_max=scaler.data_max_,
    )
    summary = {
        "name": name,
        "display": cfg["display"],
        "train": int(X_train.shape[0]),
        "test": int(X_test.shape[0]),
        "features": int(X_train.shape[1]),
        "numeric": len(cfg["numeric"]),
        "categorical": len(cfg["categorical"]),
        "pos_train": int(y_train.sum()),
        "neg_train": int((y_train == 0).sum()),
        "pos_test": int(y_test.sum()),
        "neg_test": int((y_test == 0).sum()),
    }
    print(f"  訓練 {summary['train']} 筆 / 測試 {summary['test']} 筆，"
          f"特徵 {summary['features']}（數值 {summary['numeric']} + "
          f"類別 {summary['categorical']}）")
    print(f"  標籤分佈 訓練(+/-): {summary['pos_train']}/{summary['neg_train']}，"
          f"測試(+/-): {summary['pos_test']}/{summary['neg_test']}")
    print(f"  已儲存 {out_path}")
    return summary
