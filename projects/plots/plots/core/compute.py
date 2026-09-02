from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import trange


def init_fn(det_stats, w):
    global detection_statistics, weights
    detection_statistics = det_stats
    weights = w


def compute_sv(threshold):
    mask = detection_statistics >= threshold
    mus = (weights * mask).sum(-1, keepdims=True)
    var_summands = weights * (mask - mus)
    stds = (var_summands**2).sum(-1) ** 0.5
    return mus[:, 0], stds


def sensitive_volume(detection_statistics, weights, thresholds):
    y = np.empty((len(weights), len(thresholds)))
    err = np.empty((len(weights), len(thresholds)))
    ex = ProcessPoolExecutor(
        8, initializer=init_fn, initargs=(detection_statistics, weights)
    )
    with ex:
        fs = {ex.submit(compute_sv, t): i for i, t in enumerate(thresholds)}
        for _ in trange(len(thresholds)):
            while True:
                for future in fs:
                    if future.done():
                        future.exception()
                        break
                else:
                    continue
                break

            i = fs.pop(future)
            mu, std = future.result()
            y[:, i] = mu
            err[:, i] = std
    return y, err
