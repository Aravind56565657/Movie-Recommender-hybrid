"""
Evaluation metrics for Recommender Systems
"""
import numpy as np
import pandas as pd
import logging
from math import sqrt
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def get_user_test_ratings(test_ratings, user_id):
    """Get all (movie_id, rating) for a user in test set"""
    user_data = test_ratings[test_ratings['user_id'] == user_id]
    return dict(zip(user_data['movie_id'], user_data['rating']))


def calculate_rmse(predictions, test_ratings):
    """
    Calculate RMSE on test set
    """
    # Extract predicted ratings for test user-item pairs
    y_true = []
    y_pred = []
    
    for _, row in test_ratings.iterrows():
        u = int(row['user_id']) - 1
        m = int(row['movie_id']) - 1
        
        if 0 <= u < predictions.shape[0] and 0 <= m < predictions.shape[1]:
            true_r = row['rating']
            pred_r = predictions[u, m]
            
            y_true.append(true_r)
            y_pred.append(pred_r)
            
    if not y_true:
        return 0.0
        
    return sqrt(mean_squared_error(y_true, y_pred))

def precision_recall_at_k(predictions, test_ratings, train_matrix=None, k=10, threshold=4.0):
    """
    Calculate Precision@K and Recall@K using two protocols:
    1. Global: Rank all movies (minus training)
    2. Test-Only: Rank only movies in the test set
    """
    precisions_global = []
    recalls_global = []
    precisions_test = []
    recalls_test = []
    
    test_users = test_ratings['user_id'].unique()
    
    for user_id in test_users:
        u_idx = user_id - 1
        if u_idx >= predictions.shape[0]:
            continue
            
        user_test_data = test_ratings[test_ratings['user_id'] == user_id]
        relevant_items = set(user_data['movie_id'] for _, user_data in user_test_data.iterrows() 
                             if user_data['rating'] >= threshold)
        
        if not relevant_items:
            continue
            
        # --- Protocol 1: Global ---
        user_preds = predictions[u_idx].copy()
        if train_matrix is not None:
            train_items = train_matrix.indices[train_matrix.indptr[u_idx]:train_matrix.indptr[u_idx+1]]
            user_preds[train_items] = -np.inf
            
        top_k_indices = np.argsort(user_preds)[::-1][:k]
        hits_global = 0
        for idx in top_k_indices:
            if (idx + 1) in relevant_items:
                hits_global += 1
        precisions_global.append(hits_global / k)
        recalls_global.append(hits_global / len(relevant_items))
        
        # --- Protocol 2: Test-Only ---
        movie_ids = user_test_data['movie_id'].values
        actuals = user_test_data['rating'].values
        preds = predictions[u_idx, movie_ids - 1]
        
        # Rank within test set
        ranking_indices = np.argsort(preds)[::-1][:k]
        hits_test = 0
        for idx in ranking_indices:
            if actuals[idx] >= threshold:
                hits_test += 1
        
        precisions_test.append(hits_test / min(k, len(movie_ids)))
        recalls_test.append(hits_test / len(relevant_items))
        
    return {
        'p_global': np.mean(precisions_global) if precisions_global else 0.0,
        'r_global': np.mean(recalls_global) if recalls_global else 0.0,
        'p_test': np.mean(precisions_test) if precisions_test else 0.0,
        'r_test': np.mean(recalls_test) if recalls_test else 0.0
    }

def evaluate_model(predictions, test_ratings, train_matrix=None, k_list=[10, 20], threshold=4.0):
    """
    Full evaluation suite
    """
    logger.info("Evaluating model...")
    results = {}
    
    # RMSE calculation (unchanged)
    rmse_val = calculate_rmse(predictions, test_ratings)
    results['rmse'] = rmse_val
    logger.info(f"RMSE: {rmse_val:.4f}")
    
    for k in k_list:
        metrics = precision_recall_at_k(predictions, test_ratings, train_matrix, k=k, threshold=threshold)
        
        # Calculate F1 for both protocols
        p_g, r_g = metrics['p_global'], metrics['r_global']
        f1_g = 2 * (p_g * r_g) / (p_g + r_g) if (p_g + r_g) > 0 else 0.0
        
        p_t, r_t = metrics['p_test'], metrics['r_test']
        f1_t = 2 * (p_t * r_t) / (p_t + r_t) if (p_t + r_t) > 0 else 0.0
        
        results[f'p_global@{k}'] = p_g
        results[f'r_global@{k}'] = r_g
        results[f'f1_global@{k}'] = f1_g
        
        results[f'p_test@{k}'] = p_t
        results[f'r_test@{k}'] = r_t
        results[f'f1_test@{k}'] = f1_t
        
        logger.info(f"K={k} (Global): P={p_g:.4f}, R={r_g:.4f}, F1={f1_g:.4f}")
        logger.info(f"K={k} (TestOnly): P={p_t:.4f}, R={r_t:.4f}, F1={f1_t:.4f}")
        
    return results
