import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    current_predictions = np.asarray(current_predictions, dtype=float)

    residuals = y - current_predictions

    best_mse = float("inf")
    best_predictions = np.full(len(residuals), np.mean(residuals))

    for feature_index in range(X.shape[1]):
        feature_values = np.unique(X[:, feature_index])
        thresholds = (feature_values[:-1] + feature_values[1:]) / 2

        for threshold in thresholds:
            left = (X[:, feature_index] <= threshold)
            right = ~left

            left_mean = np.mean(residuals[left])
            right_mean = np.mean(residuals[right])

            stump_predictions = np.where(left, left_mean, right_mean)

            errors = residuals - stump_predictions
            mse = np.mean(errors ** 2)

            if mse < best_mse:
                best_mse = mse
                best_predictions = stump_predictions

    updated_predictions = current_predictions + learning_rate * best_predictions

    return [round(float(value), 4) for value in updated_predictions]