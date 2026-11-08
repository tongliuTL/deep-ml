import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    X = np.asarray(X, dtype=float)

    n_samples, n_features = X.shape

    combos = [c for d in range(degree + 1)
            for c in combinations_with_replacement(range(n_features), d)]

    output = np.empty((n_samples, len(combos)))

    for j, c in enumerate(combos):
        output[:, j] = np.prod(X[:, list(c)], axis=1)

    return np.sort(output, axis=1)