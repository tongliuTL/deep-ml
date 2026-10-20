import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    positives = sum(y_true)
    negatives = len(y_true) - positives

    if positives == 0 or negatives == 0:
        return 0.0

    pairs = sorted(zip(y_scores, y_true), reverse=True)

    tp = fp = 0
    prev_tpr = prev_fpr = 0
    auc = 0.0
    i = 0

    while i < len(pairs):
        score = pairs[i][0]

        while i < len(pairs) and pairs[i][0] == score:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1

            i += 1

        tpr = tp / positives
        fpr = fp / negatives

        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)