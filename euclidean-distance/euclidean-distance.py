import numpy as np

def euclidean_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(np.sqrt(np.sum((x - y) ** 2)))