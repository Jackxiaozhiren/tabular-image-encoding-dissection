"""
igtd_encoder.py
============================================================
IGTD (Image Generator for Tabular Data) encoder — clean port.

Faithful port of the published IGTD algorithm
  Zhu et al., "Converting tabular data into images for deep learning
  with convolutional neural networks", Scientific Reports 11:11325 (2021),
  https://doi.org/10.1038/s41598-021-90923-y
from the authors' reference implementation (github.com/zhuyitan/IGTD).

Changes vs. the reference code (for reproducibility under modern
numpy/pandas): (1) removed the astropy dependency (unused in the core
path); (2) the swap-search optimizer uses a local RandomState so the
layout is deterministic given `random_state`; (3) no plotting / file
writing; images are returned as float32 tensors with values in [0,1].
The algorithm itself (feature-distance ranking, pixel-distance ranking,
greedy row/column swap search minimizing the absolute error between the
rearranged source ranking and the target ranking) is unchanged.
"""
import numpy as np
from scipy.stats import rankdata


def feature_distance_ranking(data, method="Pearson"):
    """Ranking of pairwise feature dissimilarities (num x num, symmetric)."""
    num = data.shape[1]
    if method == "Pearson":
        corr = np.corrcoef(data.T)
    elif method == "Spearman":
        from scipy.stats import spearmanr
        corr = spearmanr(data).correlation
    elif method == "Euclidean":
        from scipy.spatial.distance import squareform, pdist
        corr = squareform(pdist(data.T, metric="euclidean"))
        corr = np.max(corr) - corr
        corr = corr / np.max(corr)
    else:
        raise ValueError(f"unknown distance method {method}")
    corr = 1 - corr
    corr = np.around(corr, decimals=10)
    tril = np.tril_indices(num, k=-1)
    rank = rankdata(corr[tril])
    ranking = np.zeros((num, num))
    ranking[tril] = rank
    ranking = ranking + ranking.T
    return ranking


def matrix_distance_ranking(num_r, num_c, method="Euclidean", num=None):
    """Coordinates of `num` pixels in a num_r x num_c grid and the ranking
    of their pairwise distances (num x num, symmetric)."""
    if num is None:
        num = num_r * num_c
    coords = np.array([(r, c) for r in range(num_r) for c in range(num_c)])[:num]
    cord_dist = np.zeros((num, num))
    for i in range(num):
        if method == "Euclidean":
            cord_dist[i, :] = np.sqrt(np.square(coords[i, 0] - coords[:, 0]) +
                                      np.square(coords[i, 1] - coords[:, 1]))
        elif method == "Manhattan":
            cord_dist[i, :] = np.abs(coords[i, 0] - coords[:, 0]) + np.abs(coords[i, 1] - coords[:, 1])
    tril = np.tril_indices(num, k=-1)
    rank = rankdata(cord_dist[tril])
    ranking = np.zeros((num, num))
    ranking[tril] = rank
    ranking = ranking + ranking.T
    return (coords[:, 0].astype(np.int64), coords[:, 1].astype(np.int64)), ranking


def _igtd_absolute_error(source, target, max_step, switch_t, val_step, min_gain, random_state):
    """Greedy row/column swap search. Returns (index_record, err_record)."""
    rng = np.random.RandomState(random_state)
    source = source.copy()
    num = source.shape[0]
    tril = np.tril_indices(num, k=-1)
    index = np.arange(num)
    index_record = np.full((max_step + 1, num), np.nan)
    index_record[0, :] = index

    err_v = np.empty(num)
    for i in range(num):
        err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + \
                   np.sum(np.abs(source[(i + 1):, i] - target[(i + 1):, i]))

    step_record = -np.ones(num)
    err_record = [float(np.sum(np.abs(source[tril] - target[tril])))]
    pre_err = err_record[0]

    for s in range(max_step):
        delta = -np.ones(num) * np.inf
        idr = np.where(step_record == np.min(step_record))[0]
        ii = idr[rng.permutation(len(idr))[0]]
        for jj in range(num):
            if jj == ii:
                continue
            i, j = (ii, jj) if ii < jj else (jj, ii)
            err_ori = err_v[i] + err_v[j] - np.abs(source[j, i] - target[j, i])
            err_i = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                    np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                    np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
            err_j = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                    np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                    np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
            err_test = err_i + err_j - np.abs(source[i, j] - target[j, i])
            delta[jj] = err_ori - err_test

        delta_norm = delta / max(pre_err, 1e-12)
        id_ok = np.where(delta_norm >= switch_t)[0]
        if len(id_ok) > 0:
            jj = np.argmax(delta)
            i, j = (ii, jj) if ii < jj else (jj, ii)
            for k in range(num):
                if k < i:
                    err_v[k] = err_v[k] - np.abs(source[i, k] - target[i, k]) - np.abs(source[j, k] - target[j, k]) + \
                               np.abs(source[j, k] - target[i, k]) + np.abs(source[i, k] - target[j, k])
                elif k == i:
                    err_v[k] = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                               np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                               np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
                elif k < j:
                    err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[j, k] - target[j, k]) + \
                               np.abs(source[k, j] - target[k, i]) + np.abs(source[i, k] - target[j, k])
                elif k == j:
                    err_v[k] = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                               np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                               np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
                else:
                    err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[k, j] - target[k, j]) + \
                               np.abs(source[k, j] - target[k, i]) + np.abs(source[k, i] - target[k, j])
            ii_v = source[ii, :].copy(); jj_v = source[jj, :].copy()
            source[ii, :] = jj_v; source[jj, :] = ii_v
            ii_v = source[:, ii].copy(); jj_v = source[:, jj].copy()
            source[:, ii] = jj_v; source[:, jj] = ii_v
            err = pre_err - delta[jj]
            t = index[ii]; index[ii] = index[jj]; index[jj] = t
            step_record[ii] = s; step_record[jj] = s
        else:
            err = pre_err
            step_record[ii] = s
        err_record.append(float(err))
        index_record[s + 1, :] = index
        if s > val_step:
            rel = (err_record[-val_step - 1] - np.array(err_record[-val_step:])) / err_record[-val_step - 1]
            if np.sum(rel >= min_gain) == 0:
                break
        pre_err = err

    return index_record[:len(err_record), :].astype(int), err_record


def igtd_layout(data, num_r=8, num_c=8, max_step=400, val_step=40,
                min_gain=1e-5, switch_t=0, random_state=1):
    """Run the IGTD layout optimization on the (min-max scaled) feature matrix.

    Returns (arrangement, (rows, cols)) where `arrangement[f]` is the original
    feature index placed at pixel (rows[f], cols[f]).
    """
    source = feature_distance_ranking(data)
    (rows, cols), target = matrix_distance_ranking(num_r, num_c, "Euclidean", num=data.shape[1])
    index_record, err_record = _igtd_absolute_error(
        source, target, max_step=max_step, switch_t=switch_t,
        val_step=val_step, min_gain=min_gain, random_state=random_state)
    best = index_record[int(np.argmin(err_record)), :]
    return best, (rows, cols)


def encode_igtd(norm_data, arrangement, coords, num_r=8, num_c=8):
    """Build (n, 1, num_r, num_c) float32 images from min-max scaled [0,1] data.

    Values are placed at their optimized pixel; unoccupied pixels are 0.
    """
    rows, cols = coords
    n, p = norm_data.shape
    img = np.zeros((n, 1, num_r, num_c), dtype=np.float32)
    for f in range(p):
        img[:, 0, rows[f], cols[f]] = norm_data[:, arrangement[f]]
    return img
