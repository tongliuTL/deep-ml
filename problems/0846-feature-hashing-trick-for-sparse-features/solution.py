import hashlib

def feature_hash(features, n_features):
    """
    Apply the (signed) feature hashing trick.

    Args:
        features: dict mapping feature name (str) -> value (float).
        n_features: int, size of the output vector.

    Returns:
        list of length n_features.
    """
    vector = [0.0] * n_features

    for name, value in features.items():
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % n_features
        sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0

        vector[index] += sign * float(value)

    return vector
