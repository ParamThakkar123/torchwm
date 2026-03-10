import sys

sys.path.insert(0, ".")

import numpy as np

_original_seed = np.random.seed


def _patched_seed(seed):
    if seed is None:
        return _original_seed(seed)
    try:
        return _original_seed(seed)
    except ValueError:
        seed = abs(seed) % (2**32 - 1)
        return _original_seed(seed)


np.random.seed = _patched_seed
