import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError

    return float(np.sum(x * p))
    pass
