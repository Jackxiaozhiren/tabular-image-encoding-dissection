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
    "german": {
        "display": "UCI German Credit (Statlog)",
        "urls": {"train": f"{UCI_BASE}/statlog/german/german.data"},
        "skiprows": {},
        "delimiter": r"\s+",
        "has_header": False,
        "column_names": [
            "status", "duration", "credit_history", "purpose", "credit_amount",
            "savings", "employment", "installment_rate", "personal_status",
            "other_debtors", "residence_since", "property", "age",
            "other_installment_plans", "housing", "existing_credits", "job",
            "dependents", "telephone", "foreign_worker", "label",
        ],
        "numeric": ["duration", "credit_amount", "installment_rate",
                    "residence_since", "age", "existing_credits", "dependents"],
        "categorical": ["status", "credit_history", "purpose", "savings",
                        "employment", "personal_status", "other_debtors",
                        "property", "other_installment_plans", "housing",
                        "job", "telephone", "foreign_worker"],
        "label": "label",
        "label_map": {"2": 1},          # 正例 = 違約（bad credit，少數類 ~30%）
        "has_test": False,
        "test_ratio": 0.25,
    },
    "telco": {
        "display": "Telco Customer Churn",
        "urls": {"train": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"},
        "skiprows": {},
        "has_header": True,
        "na_values": [" "],
        "column_names": [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
        ],
        "numeric": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        "categorical": ["gender", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod"],
        "label": "Churn",
        "label_map": {"Yes": 1},
        "drop_columns": ["customerID"],  # 無意義識別碼
        "has_test": False,
        "test_ratio": 0.25,
        "numeric_impute": True,          # TotalCharges 含少量空值
    },
    "sick": {
        "display": "UCI Sick (Thyroid)",
        "urls": {"train": f"{UCI_BASE}/thyroid-disease/sick.data"},
        "skiprows": {},
        "has_header": False,
        "column_names": [
            "age", "sex", "on_thyroxine", "query_on_thyroxine",
            "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
            "i131_treatment", "query_hypothyroid", "query_hyperthyroid",
            "lithium", "goitre", "tumor", "hypopituitary", "psych",
            "tsh_measured", "tsh", "t3_measured", "t3", "tt4_measured", "tt4",
            "t4u_measured", "t4u", "fti_measured", "fti", "tbg_measured", "tbg",
            "referral_source", "class",
        ],
        "numeric": ["age", "tsh", "t3", "tt4", "t4u", "fti"],
        "categorical": ["sex", "on_thyroxine", "query_on_thyroxine",
                        "on_antithyroid_medication", "sick", "pregnant",
                        "thyroid_surgery", "i131_treatment", "query_hypothyroid",
                        "query_hyperthyroid", "lithium", "goitre", "tumor",
                        "hypopituitary", "psych", "tsh_measured", "t3_measured",
                        "tt4_measured", "t4u_measured", "fti_measured",
                        "referral_source"],
        "label": "class",
        "label_map": {"sick": 1},        # 正例 = sick（少數類 ~6%）
        "label_clean": "|",              # 原始標籤 "sick.|NNNN" -> "sick"
        "drop_columns": ["tbg", "tbg_measured"],  # 全缺失 / 全常數
        "has_test": False,
        "test_ratio": 0.25,
        "numeric_impute": True,          # 多個數值欄含 "?"
    },
    # ------------------------------------------------------------------
    # OpenML 資料集（v2 擴充，2026-08-11）。經 openml 套件下載、快取於 data/raw_*.csv。
    # 全數值資料集（segment/spambase/magic/vehicle）走「Wine 模式」：M1c ≡ M1、
    # M3-noG ≡ M3-full，由 08 統計自動排除對應對照；其餘對照（B 通道/重排/影像形式/
    # 樹 vs 影像）全部有效。
    # 多類別資料集（cmc/segment/vehicle）以「眾數類別 vs 其餘」二值化（binarize_majority）。
    # ------------------------------------------------------------------
    "australian": {
        "display": "Australian Credit Approval (Statlog)",
        "openml_id": 40981,
        "column_names": [f"A{i}" for i in range(1, 16)],
        "numeric": ["A2", "A3", "A7", "A10", "A13", "A14"],
        "categorical": ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"],
        "label": "A15",
        "label_map": {"1": 1},
        "has_test": False,
        "test_ratio": 0.25,
    },
    "cmc": {
        "display": "Contraceptive Method Choice (cmc)",
        "openml_id": 23,
        "column_names": ["Wifes_age", "Wifes_education", "Husbands_education",
                         "Wifes_religion", "Wifes_now_working",
                         "Husbands_occupation", "Standard-of-living_index",
                         "Media_exposure", "Number_of_children_ever_born",
                         "Contraceptive_method_used"],
        "numeric": ["Wifes_age", "Number_of_children_ever_born"],
        "categorical": ["Wifes_education", "Husbands_education", "Wifes_religion",
                        "Wifes_now_working", "Husbands_occupation",
                        "Standard-of-living_index", "Media_exposure"],
        "label": "Contraceptive_method_used",
        "binarize_majority": True,       # 3 類 → 眾數 vs 其餘
        "has_test": False,
        "test_ratio": 0.25,
    },
    "ilpd": {
        "display": "Indian Liver Patient Dataset (ILPD)",
        "openml_id": 1480,
        "column_names": [f"V{i}" for i in range(1, 11)] + ["Class"],
        "numeric": [f"V{i}" for i in [1, 3, 4, 5, 6, 7, 8, 9, 10]],
        "categorical": ["V2"],
        "label": "Class",
        "label_map": {"1": 1},           # 1 = 有肝病
        "has_test": False,
        "test_ratio": 0.25,
    },
    "segment": {
        "display": "Image Segmentation (segment)",
        "openml_id": 36,
        "column_names": ["region-centroid-col", "region-centroid-row",
                         "region-pixel-count", "short-line-density-5",
                         "short-line-density-2", "vedge-mean", "vegde-sd",
                         "hedge-mean", "hedge-sd", "intensity-mean",
                         "rawred-mean", "rawblue-mean", "rawgreen-mean",
                         "exred-mean", "exblue-mean", "exgreen-mean",
                         "value-mean", "saturation-mean", "hue-mean", "class"],
        "numeric": ["region-centroid-col", "region-centroid-row",
                    "region-pixel-count", "short-line-density-5",
                    "short-line-density-2", "vedge-mean", "vegde-sd",
                    "hedge-mean", "hedge-sd", "intensity-mean",
                    "rawred-mean", "rawblue-mean", "rawgreen-mean",
                    "exred-mean", "exblue-mean", "exgreen-mean",
                    "value-mean", "saturation-mean", "hue-mean"],
        "categorical": [],
        "label": "class",
        "binarize_majority": True,       # 7 類 → 眾數 vs 其餘
        "has_test": False,
        "test_ratio": 0.25,
    },
    "vehicle": {
        "display": "Vehicle Silhouettes",
        "openml_id": 54,
        "column_names": ["COMPACTNESS", "CIRCULARITY", "DISTANCE_CIRCULARITY",
                         "RADIUS_RATIO", "PR.AXIS_ASPECT_RATIO",
                         "MAX.LENGTH_ASPECT_RATIO", "SCATTER_RATIO",
                         "ELONGATEDNESS", "PR.AXIS_RECTANGULARITY",
                         "MAX.LENGTH_RECTANGULARITY", "SCALED_VARIANCE_MAJOR",
                         "SCALED_VARIANCE_MINOR", "SCALED_RADIUS_OF_GYRATION",
                         "SKEWNESS_ABOUT_MAJOR", "SKEWNESS_ABOUT_MINOR",
                         "KURTOSIS_ABOUT_MAJOR", "KURTOSIS_ABOUT_MINOR",
                         "HOLLOWS_RATIO", "Class"],
        "numeric": ["COMPACTNESS", "CIRCULARITY", "DISTANCE_CIRCULARITY",
                    "RADIUS_RATIO", "PR.AXIS_ASPECT_RATIO",
                    "MAX.LENGTH_ASPECT_RATIO", "SCATTER_RATIO",
                    "ELONGATEDNESS", "PR.AXIS_RECTANGULARITY",
                    "MAX.LENGTH_RECTANGULARITY", "SCALED_VARIANCE_MAJOR",
                    "SCALED_VARIANCE_MINOR", "SCALED_RADIUS_OF_GYRATION",
                    "SKEWNESS_ABOUT_MAJOR", "SKEWNESS_ABOUT_MINOR",
                    "KURTOSIS_ABOUT_MAJOR", "KURTOSIS_ABOUT_MINOR",
                    "HOLLOWS_RATIO"],
        "categorical": [],
        "label": "Class",
        "binarize_majority": True,       # 4 類 → 眾數 vs 其餘
        "has_test": False,
        "test_ratio": 0.25,
    },
    "spambase": {
        "display": "Spambase (Spam E-mail)",
        "openml_id": 44,
        "column_names": ["word_freq_make", "word_freq_address", "word_freq_all",
                         "word_freq_3d", "word_freq_our", "word_freq_over",
                         "word_freq_remove", "word_freq_internet",
                         "word_freq_order", "word_freq_mail", "word_freq_receive",
                         "word_freq_will", "word_freq_people", "word_freq_report",
                         "word_freq_addresses", "word_freq_free", "word_freq_business",
                         "word_freq_email", "word_freq_you", "word_freq_credit",
                         "word_freq_your", "word_freq_font", "word_freq_000",
                         "word_freq_money", "word_freq_hp", "word_freq_hpl",
                         "word_freq_george", "word_freq_650", "word_freq_lab",
                         "word_freq_labs", "word_freq_telnet", "word_freq_857",
                         "word_freq_data", "word_freq_415", "word_freq_85",
                         "word_freq_technology", "word_freq_1999", "word_freq_parts",
                         "word_freq_pm", "word_freq_direct", "word_freq_cs",
                         "word_freq_meeting", "word_freq_original", "word_freq_project",
                         "word_freq_re", "word_freq_edu", "word_freq_table",
                         "word_freq_conference", "char_freq_38", "char_freq_40",
                         "char_freq_91", "char_freq_33", "char_freq_36",
                         "char_freq_35", "capital_run_length_average",
                         "capital_run_length_longest", "capital_run_length_total",
                         "class"],
        "numeric": ["word_freq_make", "word_freq_address", "word_freq_all",
                    "word_freq_3d", "word_freq_our", "word_freq_over",
                    "word_freq_remove", "word_freq_internet",
                    "word_freq_order", "word_freq_mail", "word_freq_receive",
                    "word_freq_will", "word_freq_people", "word_freq_report",
                    "word_freq_addresses", "word_freq_free", "word_freq_business",
                    "word_freq_email", "word_freq_you", "word_freq_credit",
                    "word_freq_your", "word_freq_font", "word_freq_000",
                    "word_freq_money", "word_freq_hp", "word_freq_hpl",
                    "word_freq_george", "word_freq_650", "word_freq_lab",
                    "word_freq_labs", "word_freq_telnet", "word_freq_857",
                    "word_freq_data", "word_freq_415", "word_freq_85",
                    "word_freq_technology", "word_freq_1999", "word_freq_parts",
                    "word_freq_pm", "word_freq_direct", "word_freq_cs",
                    "word_freq_meeting", "word_freq_original", "word_freq_project",
                    "word_freq_re", "word_freq_edu", "word_freq_table",
                    "word_freq_conference", "char_freq_38", "char_freq_40",
                    "char_freq_91", "char_freq_33", "char_freq_36",
                    "char_freq_35", "capital_run_length_average",
                    "capital_run_length_longest", "capital_run_length_total"],
        "categorical": [],
        "label": "class",
        "label_map": {"1": 1},           # 1 = spam
        "has_test": False,
        "test_ratio": 0.25,
    },
    "magic": {
        "display": "MAGIC Gamma Telescope",
        "openml_id": 1120,
        "column_names": ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                         "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
                         "class"],
        "numeric": ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"],
        "categorical": [],
        "label": "class",
        "label_map": {"g": 1},           # g = gamma（訊號）
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
    """依資料集設定讀取資料（支援 CSV / zip 內 CSV / Excel / 空白分隔）。"""
    url = str(path_or_url).lower()
    na_values = cfg.get("na_values")
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
                               skiprows=skiprows, na_values=na_values)
    if url.endswith((".xls", ".xlsx")):
        return pd.read_excel(path_or_url, header=0,
                             skiprows=skiprows)
    sep = cfg.get("delimiter", ",")
    if sep in (r"\s+", "\\s+"):
        return pd.read_csv(path_or_url, header=0 if cfg.get("has_header") else None,
                           sep=r"\s+", engine="python", skiprows=skiprows,
                           na_values=na_values)
    return pd.read_csv(path_or_url, header=0 if cfg.get("has_header") else None,
                       delimiter=sep, skiprows=skiprows, na_values=na_values)


def _load_openml(name: str, cfg: dict):
    """從 OpenML 抓取資料集（僅首次；快取於 data/raw_{name}.csv）。
    回傳 (df, None)，欄名依 cfg['column_names'] 按位置重命名（與 UCI 資料流一致）。"""
    import openml
    try:
        openml.config.set_root_cache_directory(
            os.path.join(BASE_DIR, ".openml_cache"))
    except Exception:
        pass
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = openml.datasets.get_dataset(cfg["openml_id"], download_data=True)
        df = d.get_data(dataset_format="dataframe")[0]
    if len(df.columns) == len(cfg["column_names"]):
        df.columns = cfg["column_names"]
    return df, None


def load_raw(name: str):
    """
    下載資料集原始資料（有快取則讀取快取）。
    adult 回傳 (train_df, test_df)；heart/wine 回傳 (df, None)。
    OpenML 資料集回傳 (df, None)。
    """
    cfg = DATASETS[name]
    if cfg.get("openml_id"):
        cache_file = os.path.join(DATA_DIR, f"raw_{name}.csv")
        if not os.path.exists(cache_file):
            print(f"  下載 {cfg['display']}（OpenML id={cfg['openml_id']}）...")
            df, _ = _load_openml(name, cfg)
            df.to_csv(cache_file, index=False, header=False)
            return df, None
        else:
            df = pd.read_csv(cache_file, header=None)
            if len(df.columns) == len(cfg["column_names"]):
                df.columns = cfg["column_names"]
            return df, None
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
        if cfg.get("label_clean"):      # 例：sick 標籤為 "sick.|NNNN" -> "sick"
            df[cfg["label"]] = (df[cfg["label"]].astype(str)
                                .str.split(cfg["label_clean"]).str[0].str.strip("."))
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.rstrip(".")
    df = df.replace("?", np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _binarize_label(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """標籤二值化：adult/bank 依 label_map；heart/wine 依 label_ge 門檻；
    credit 已為 0/1 直接取整；多類別資料集可依 binarize_majority（眾數 vs 其餘）。"""
    if cfg.get("binarize_majority"):
        vals = df[cfg["label"]].astype(str).values
        majority = pd.Series(vals).value_counts().idxmax()
        return (vals == majority).astype(int)
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


def _encode_cat_split(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      cfg: dict) -> tuple:
    """洩漏防護版類別編碼：眾數與 LabelEncoder 皆僅在訓練集擬合。

    測試集未見類別對應到訓練集眾數的代碼（避免 -1 破壞後續 one-hot）。
    回傳 (X_train_cat, X_test_cat)，皆 (n, n_cat) label-encoded。
    """
    n_cat = len(cfg["categorical"])
    out_tr = np.zeros((len(train_df), n_cat), dtype=float)
    out_te = np.zeros((len(test_df), n_cat), dtype=float)
    for j, col in enumerate(cfg["categorical"]):
        mode_val = train_df[col].mode()[0]
        s_tr = train_df[col].fillna(mode_val).astype(str)
        s_te = test_df[col].fillna(mode_val).astype(str)
        enc = LabelEncoder()
        enc.fit(s_tr)
        classes = set(enc.classes_)
        mode_code = enc.transform([str(mode_val)])[0]
        out_tr[:, j] = enc.transform(s_tr)
        out_te[:, j] = np.array(
            [enc.transform([x])[0] if x in classes else mode_code for x in s_te])
    return out_tr, out_te


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

        # 類別編碼：眾數填充與 LabelEncoder 皆僅在訓練集擬合（洩漏防護），
        # 測試集未見類別映射到訓練集眾數的代碼
        X_train_cat, X_test_cat = _encode_cat_split(train, test, cfg)

        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
    else:
        # -------- heart/wine/bank/credit：先切分，再以訓練集擬合統計 --------
        df = _clean(train, cfg["column_names"], cfg)
        y = _binarize_label(df, cfg)
        # 先做分層切分：確保 Min-Max / 眾數 / LabelEncoder 全部僅用訓練集
        # （修正舊版「先縮放後切分」的洩漏：scaler 曾對含測試行的全資料 fit）
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=cfg.get("test_ratio", 0.25),
            stratify=y, random_state=seed)
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 數值中位數插補：僅以訓練集擬合（sick/telco 等含缺失數值值的資料集）
        train_num = train_df[cfg["numeric"]].astype(float)
        test_num = test_df[cfg["numeric"]].astype(float)
        if cfg.get("numeric_impute"):
            med = train_num.median()
            train_num = train_num.fillna(med)
            test_num = test_num.fillna(med)

        scaler = MinMaxScaler()
        X_train_num = scaler.fit_transform(train_num)
        X_test_num = scaler.transform(test_num)
        X_train_cat, X_test_cat = _encode_cat_split(train_df, test_df, cfg)

        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])

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
