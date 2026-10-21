import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    # Your code here
    n = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins

        samples = []

        for label, prob in zip(y_true, y_prob):
            if lower < prob <= upper:
                samples.append((label, prob))
            elif i == 0 and prob == 0:
                samples.append((label, prob))

        if not samples:
            continue

        accuracy = sum(label for label, _ in samples) / len(samples)
        confidence = sum(prob for _, prob in samples) / len(samples)
        weight = len(samples) / n

        ece += weight * abs(accuracy - confidence)

    return round(ece, 3)