from __future__ import annotations

import math
from typing import Iterable, List, Dict

import numpy as np


def compute_aggregate_metrics(per_seed_means: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(per_seed_means), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "iqm": 0.0, "num_seeds": 0}

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    # IQM: mean of 25/50/75 percentiles as simple robust metric (not full iqm implementation)
    iqm = float(np.mean(np.percentile(arr, [25, 50, 75])))

    return {
        "mean": mean,
        "median": median,
        "iqm": iqm,
        "num_seeds": int(arr.size),
    }


def bootstrap_ci(values: List[float], num_samples: int = 1000, alpha: float = 0.05):
    """Compute simple bootstrap 1-alpha CI on the mean."""
    if not values:
        return (0.0, 0.0)
    vals = np.array(values)
    n = vals.size
    means = []
    for _ in range(num_samples):
        sample = np.random.choice(vals, size=n, replace=True)
        means.append(sample.mean())
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper
