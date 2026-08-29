"""
tabular_to_image.py
============================================================
表格資料 -> 圖像編碼共用模組（供 02/03 腳本匯入）

三種編碼方法：
  M1 特徵列熱圖（Feature-Row Heatmap，灰度）
     每個數值特徵佔一列，列上所有像素填該特徵歸一化後的值；
     保留「特徵 x 值」的整潔對應，CNN 可用縱向捲積捕捉特徵關係。

  M2 相關性排序 2D 拼貼（Correlation-Sorted 2D Grid，IGTD 風格）
     依特徵與目標的絕對相關係數降序排序，重要特徵重複填入 NxN 方格；
     縮短重要特徵在圖面上的距離，方便捲積核同時看到它們。

  M3 多通道編碼（Multi-Channel Encoding，本研究提出）
     R 通道：數值特徵熱圖（同 M1）
     G 通道：類別特徵 one-hot 區塊（每類別特徵一列，類別索引處為 1）
     B 通道：依特徵重要度（weight_map）重排的數值熱圖，即「權重重排」：
             重要特徵對應列越靠上，越早被捲積核掃過並享有更大感受野。

所有編碼回傳 np.float32 的 (通道, 高, 寬) 張量，可直接餵給 CNN。
============================================================
"""
import numpy as np


def minmax_norm(num_matrix: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """逐欄 Min-Max 歸一化到 [0,1]。mins/maxs 必須來自訓練集（避免資料洩漏）。"""
    return np.clip((num_matrix - mins) / (maxs - mins + 1e-9), 0.0, 1.0)


def _row_heatmap(values: np.ndarray, width: int) -> np.ndarray:
    """把一列特徵值橫向重複 width 次，形成 (len(values), width) 的熱圖矩陣。"""
    return np.tile(np.asarray(values, dtype=float), (width, 1)).T


def encode_m1(num_vec: np.ndarray, width: int = 64) -> np.ndarray:
    """M1：灰度特徵列熱圖。回傳 (1, n_features, width)。"""
    img = _row_heatmap(num_vec, width)
    return img[None, :, :].astype(np.float32)


def encode_m2(num_vec: np.ndarray, corr: np.ndarray, grid: int = 4) -> np.ndarray:
    """
    M2：依 |corr|（與目標的絕對相關係數）降序排序特徵，
    並將重要特徵重複填滿 grid x grid 方格（權重重排的雛形）。
    回傳 (1, grid, grid)。
    """
    order = np.argsort(-np.abs(np.asarray(corr, dtype=float)))
    ordered = np.clip(np.asarray(num_vec, dtype=float)[order], 0.0, 1.0)
    n_cells = grid * grid
    reps = int(np.ceil(n_cells / max(len(ordered), 1)))   # 每個特徵重複的次數
    flat = np.repeat(ordered, reps)[:n_cells]              # 重複後截斷到 n_cells
    flat = np.pad(flat, (0, max(0, n_cells - flat.size)))[:n_cells]
    return flat.reshape(1, grid, grid).astype(np.float32)


def encode_m1_cat(num_vec: np.ndarray, cat_oh: list, width: int = 64,
                  height: int = 16) -> np.ndarray:
    """
    M1c：單通道版「類別包容式編碼」（用於消融驗證）。
      與 M3 容納相同的資訊（數值特徵 + 類別 one-hot），但全部垂直拼接於
      同一張灰度圖，而非分成 RGB 三個通道：
        [前 n_num 列] 數值特徵熱圖
        [其後各列]    類別特徵 one-hot 區塊
      若 M1c 與 M3 效能相當，代表「多通道分離」並非關鍵，
      納入類別資訊才是效能增益的來源。
    回傳 (1, height, width)。
    """
    img = np.zeros((height, width), dtype=np.float32)
    num_vec = np.clip(np.asarray(num_vec, dtype=float), 0.0, 1.0)
    n_num = min(len(num_vec), height)
    img[:n_num] = _row_heatmap(num_vec[:n_num], width)
    for j, oh in enumerate(cat_oh):
        r = n_num + j
        if r >= height:
            break
        n = min(len(oh), width)
        img[r, :n] = np.clip(np.asarray(oh[:n], dtype=float), 0.0, 1.0)
    return img[None, :, :].astype(np.float32)


def encode_m3(num_vec: np.ndarray, cat_oh: list, width: int = 64, height: int = 16,
              weight_map: np.ndarray = None, use_cat: bool = True,
              zero_b: bool = False) -> np.ndarray:
    """
    M3：三通道彩色編碼（本研究提出）。
      R 通道：數值特徵熱圖（前 height 列）；
      G 通道：類別特徵 one-hot 區塊（每類別特徵一列；use_cat=False 時置零，
             用於消融實驗驗證 G 通道的貢獻）；
      B 通道：依 weight_map（特徵重要度）降序重排的數值熱圖（權重重排；
             weight_map=None 時等同 R 通道原順序，即不重排，用於消融；
             zero_b=True 時整個置零，形成純兩通道 R+G 控制，使 M1c-vs-M3-RG
             乾淨隔離「通道分離」效應）。
    cat_oh：長度 = 類別特徵數的 list，每個元素是該特徵的 one-hot 向量。
    weight_map：長度 = 數值特徵數的權重陣列（例如 XGBoost 特徵重要度 / SHAP）。
    回傳 (3, height, width)。
    """
    R = np.zeros((height, width), dtype=np.float32)
    G = np.zeros((height, width), dtype=np.float32)
    B = np.zeros((height, width), dtype=np.float32)

    num_vec = np.clip(np.asarray(num_vec, dtype=float), 0.0, 1.0)

    # --- R 通道：數值特徵熱圖 ---
    n_num = min(len(num_vec), height)
    R[:n_num] = _row_heatmap(num_vec[:n_num], width)

    # --- G 通道：類別特徵 one-hot 區塊 ---
    if use_cat:
        for j, oh in enumerate(cat_oh[:height]):
            n = min(len(oh), width)
            G[j, :n] = np.clip(np.asarray(oh[:n], dtype=float), 0.0, 1.0)

    # --- B 通道：權重重排（重要度越高越靠上；None 則不重排；zero_b 則置零） ---
    if not zero_b:
        if weight_map is not None:
            w = np.asarray(weight_map, dtype=float)
            order = np.argsort(-w)
            n_rep = min(len(order), height)
            B[:n_rep] = _row_heatmap(num_vec[order[:n_rep]], width)
        else:
            B[:n_num] = _row_heatmap(num_vec[:n_num], width)

    return np.stack([R, G, B], axis=0)  # (3, H, W)
