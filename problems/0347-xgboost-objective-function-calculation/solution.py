import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    def sums(indices):
        return np.sum(gradients[indices]), np.sum(hessians[indices])

    left_g, left_h = sums(left_indices)
    right_g, right_h = sums(right_indices)
    parent_g = left_g + right_g
    parent_h = left_h + right_h

    left_weight = -left_g / (left_h + lambda_reg)
    right_weight = -right_g / (right_h + lambda_reg)

    left_score = left_g ** 2 / (left_h + lambda_reg)
    right_score = right_g ** 2 / (right_h + lambda_reg)
    parent_score = parent_g ** 2 / (parent_h + lambda_reg)

    gain = (left_score + right_score - parent_score) * 0.5 - gamma
    
    return {
        'left_weight': round(float(left_weight), 4),
        'right_weight': round(float(right_weight), 4),
        'gain': round(float(gain), 4),
    }