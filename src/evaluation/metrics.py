"""
Basic metrics: RMSE for ratings
"""
import numpy as np

def rmse(pred_matrix, rating_matrix):
    """
    Compute RMSE on non-zero entries of rating_matrix (CSR)

    Args:
        pred_matrix: np.array (n_users, n_movies)
        rating_matrix: CSR matrix

    Returns:
        float RMSE
    """
    coo = rating_matrix.tocoo()
    preds = pred_matrix[coo.row, coo.col]
    rmse_val = np.sqrt(np.mean((coo.data - preds) ** 2))
    return rmse_val
