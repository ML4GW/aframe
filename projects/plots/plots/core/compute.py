import numpy as np


def sensitive_volume(detection_statistics, weights, thresholds):
    """Mean and standard error of sensitive volume at each threshold.

    Args:
        detection_statistics: `(N,)` array of injection detection stats.
        weights: `(C, N)` array of per-injection weights, one row per
            mass combo.
        thresholds: `(T,)` array of detection statistic thresholds.

    Returns:
        `(y, err)`, each `(C, T)`.
    """
    order = np.argsort(detection_statistics)
    ds_sorted = detection_statistics[order]
    weights_sorted = weights[:, order]

    # Compute the reverse cumulative sums of the weights and the
    # weights squared. Because the weights are sorted by the
    # detection statistics, this avoids needing to compute a mask
    # for each threshold.
    n = len(ds_sorted)
    rev_sum_w = np.zeros((weights.shape[0], n + 1))
    rev_sum_w2 = np.zeros_like(rev_sum_w)
    rev_sum_w[:, :-1] = np.cumsum(weights_sorted[:, ::-1], axis=-1)[:, ::-1]
    rev_sum_w2[:, :-1] = np.cumsum(weights_sorted[:, ::-1] ** 2, axis=-1)[
        :, ::-1
    ]

    # idxs is the indices of the first detection statistic greater than
    # each threshold.
    idxs = np.searchsorted(ds_sorted, thresholds)
    mu = rev_sum_w[:, idxs]
    var = (1 - 2 * mu) * rev_sum_w2[:, idxs] + mu**2 * rev_sum_w2[:, :1]
    err = np.sqrt(np.maximum(var, 0))
    return mu, err
