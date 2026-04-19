"""
Step 1: ALS (Alternating Least Squares) Matrix Factorization
Decomposes sparse rating matrix into user and movie latent features
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from tqdm import tqdm
import logging
import sys

logger = logging.getLogger(__name__)

class ALSRecommender:
    """
    ALS-based collaborative filtering recommender
    Implements Step 1 from the paper
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dict with 'factors', 'regularization', 'iterations', etc.
        """
        self.config = config
        self.user_factors = None  # U matrix (n_users x k)
        self.item_factors = None  # M matrix (n_movies x k)
        self.user_biases = None   # b_u (n_users,)
        self.item_biases = None   # b_i (n_movies,)
        self.global_mean = 0.0    # mu
        self.trained = False
        
    def fit(self, rating_matrix, init_user_factors=None, init_item_factors=None):
        """
        Train ALS model using Explicit Feedback (since ratings are 1-5)
        
        Args:
            rating_matrix: Sparse CSR matrix (users x movies)
            init_user_factors: Optional (n_users, n_factors) array for warm-start
            init_item_factors: Optional (n_items, n_factors) array for warm-start
        """
        logger.info("="*60)
        logger.info("STEP 1: TRAINING ALS MODEL (EXPLICIT FEEDBACK)")
        logger.info("="*60)
        
        n_users, n_items = rating_matrix.shape
        n_factors = self.config['factors']
        reg = self.config['regularization']
        bias_reg = self.config.get('bias_regularization', reg)
        n_iters = self.config['iterations']
        
        # Compute global mean
        self.global_mean = rating_matrix.data.mean()
        logger.info(f"Global mean rating: {self.global_mean:.4f}")
        
        # Initialize biases to zero
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Initialize latent factors
        np.random.seed(self.config.get('random_state', 42))
        
        # User Factors
        if init_user_factors is not None:
            if init_user_factors.shape != (n_users, n_factors):
                logger.warning(f"Shape mismatch: init_user_factors {init_user_factors.shape} vs {(n_users, n_factors)}. Using random.")
                self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))
            else:
                logger.info("Warm-starting user factors.")
                self.user_factors = init_user_factors.copy()
        else:
            self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))
            
        # Item Factors
        if init_item_factors is not None:
            if init_item_factors.shape != (n_items, n_factors):
                logger.warning(f"Shape mismatch: init_item_factors {init_item_factors.shape} vs {(n_items, n_factors)}. Using random.")
                self.item_factors = np.random.normal(scale=0.1, size=(n_items, n_factors))
            else:
                logger.info("Warm-starting item factors.")
                self.item_factors = init_item_factors.copy()
        else:
            self.item_factors = np.random.normal(scale=0.1, size=(n_items, n_factors))
        
        # Precompute I (identity matrix) once
        reg_I = reg * np.eye(n_factors)
        
        csc_ratings = rating_matrix.tocsc()
        
        print(f"Starting Biased ALS training: {n_iters} iterations")

        best_rmse = np.inf
        no_improve_count = 0
        early_stop_patience = 3   # stop if RMSE doesn't improve for 3 checks (every 5 iters)
        
        for i in range(n_iters):
            logger.info(f"Iteration {i+1}/{n_iters}")
            print(f"Iteration {i+1}/{n_iters}")
            
            # --- Update User Factors & Biases ---
            for u in tqdm(range(n_users), desc=f"Iter {i+1} Users", mininterval=1.0):
                start = rating_matrix.indptr[u]
                end = rating_matrix.indptr[u+1]
                if start == end: continue
                
                item_indices = rating_matrix.indices[start:end]
                ratings = rating_matrix.data[start:end]
                
                # Residual calculation for user bias and factors
                # r_ui - mu - b_i
                residuals = ratings - self.global_mean - self.item_biases[item_indices]
                
                # Update user bias: b_u = sum(residuals - p_u . q_i) / (n_rated + reg)
                # But typically we update bias and factors alternatively or together.
                # Accurate ALS with biases:
                # Update b_u:
                dot_products = self.item_factors[item_indices] @ self.user_factors[u]
                self.user_biases[u] = np.sum(residuals - dot_products) / (len(item_indices) + bias_reg)
                
                # Update user factors P_u:
                # Update P_u: solve (Y_u^T Y_u + reg*n_u*I) p_u = Y_u^T * residuals_minus_bias
                recent_residuals = residuals - self.user_biases[u]
                Y_u = self.item_factors[item_indices]
                A = Y_u.T @ Y_u + (reg * len(item_indices) * np.eye(n_factors))
                b = Y_u.T @ recent_residuals
                try:
                    self.user_factors[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass

            # -------------------------------------------------------
            # STEP B: Update Item Factors
            # -------------------------------------------------------
            for i_idx in tqdm(range(n_items), desc=f"Iter {i+1} Items", mininterval=1.0):
                indices = csc_ratings[:, i_idx].indices
                if len(indices) == 0:
                    continue
                
                ratings = csc_ratings[:, i_idx].data - self.global_mean - self.user_biases[indices] - self.item_biases[i_idx]
                
                U_i = self.user_factors[indices]
                A = U_i.T @ U_i + (reg * len(indices) * np.eye(n_factors))
                V = U_i.T @ ratings
                try:
                    self.item_factors[i_idx] = np.linalg.solve(A, V)
                except np.linalg.LinAlgError:
                    pass

            # Update User Biases
            for u_idx in tqdm(range(n_users), desc=f"Iter {i+1} User Biases", mininterval=1.0):
                indices = rating_matrix[u_idx].indices
                if len(indices) == 0:
                    continue
                ratings_for_bias = rating_matrix[u_idx].data - self.global_mean - self.item_biases[indices] - (self.item_factors[indices] @ self.user_factors[u_idx])
                self.user_biases[u_idx] = np.sum(ratings_for_bias) / (len(indices) + bias_reg)

            # Update Item Biases
            for i_idx in tqdm(range(n_items), desc=f"Iter {i+1} Item Biases", mininterval=1.0):
                indices = csc_ratings[:, i_idx].indices
                if len(indices) == 0:
                    continue
                ratings_for_bias = csc_ratings[:, i_idx].data - self.global_mean - self.user_biases[indices] - (self.user_factors[indices] @ self.item_factors[i_idx])
                self.item_biases[i_idx] = np.sum(ratings_for_bias) / (len(indices) + bias_reg)
            
            # Compute training RMSE
            if (i + 1) % 5 == 0 or (i + 1) == n_iters:
                train_preds = self.predict_train(rating_matrix)
                train_rmse = np.sqrt(np.mean((rating_matrix.data - train_preds)**2))
                logger.info(f"Iteration {i+1} Training RMSE: {train_rmse:.4f}")
                print(f"Iteration {i+1} Training RMSE: {train_rmse:.4f}")
                
                # Early stopping: halt if RMSE hasn't improved
                if train_rmse < best_rmse - 1e-4:
                    best_rmse = train_rmse
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= early_stop_patience:
                        logger.info(f"Early stopping at iteration {i+1} — RMSE plateaued at {best_rmse:.4f}")
                        print(f"Early stopping at iteration {i+1} — RMSE plateaued at {best_rmse:.4f}")
                        break
                
        self.trained = True
        logger.info("ALS TRAINING COMPLETE")
        return self

    
    def predict(self, user_id=None, movie_id=None):
        """
        Predict ratings and clip to 1-5 range
        
        Args:
            user_id: Single user ID or None for all users
            movie_id: Single movie ID or None for all movies
            
        Returns:
            Predicted rating(s) or full prediction matrix
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if user_id is not None and movie_id is not None:
            # Predict single rating (user_id and movie_id are 1-indexed)
            user_idx = user_id - 1
            movie_idx = movie_id - 1
            rating = self.global_mean + self.user_biases[user_idx] + self.item_biases[movie_idx] + \
                     np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
            return float(np.clip(rating, 1.0, 5.0))
        
        elif user_id is not None:
            # Predict all movies for a user
            user_idx = user_id - 1
            user_vec = self.user_factors[user_idx]
            bias_offset = self.global_mean + self.user_biases[user_idx]
            ratings = self.item_factors @ user_vec + bias_offset + self.item_biases
            return np.clip(ratings, 1.0, 5.0)
        
        else:
            # Predict full matrix R_hat
            logger.info("Computing full prediction matrix...")
            # Broadcasting biases
            prediction_matrix = self.global_mean + \
                                self.user_biases[:, np.newaxis] + \
                                self.item_biases[np.newaxis, :] + \
                                (self.user_factors @ self.item_factors.T)
            return np.clip(prediction_matrix, 1.0, 5.0)

    def predict_train(self, rating_matrix):
        """
        Efficiently predict ratings for only observed user-item pairs in training
        """
        user_indices, item_indices = rating_matrix.nonzero()
        n_samples = len(user_indices)
        preds = np.zeros(n_samples, dtype=np.float32)
        
        chunk_size = 50000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            u_chunk = user_indices[start:end]
            i_chunk = item_indices[start:end]
            
            dot_prod = np.sum(self.user_factors[u_chunk] * self.item_factors[i_chunk], axis=1)
            preds[start:end] = (self.global_mean 
                                + self.user_biases[u_chunk] 
                                + self.item_biases[i_chunk] 
                                + dot_prod)
                                
        return np.clip(preds, 1.0, 5.0)
    
    def get_user_factors(self):
        """Get user latent factors (U matrix)"""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        return self.user_factors
    
    def get_item_factors(self):
        """Get item latent factors (M matrix)"""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        return self.item_factors

    def get_baseline_info(self):
        """Get global mean and item biases for other models to use as baselines"""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        return {
            'global_mean': self.global_mean,
            'item_biases': self.item_biases
        }
    
    def recommend(self, user_id, n=10, filter_already_rated=True, 
              rating_matrix=None):
        """
        Recommend top-N movies for a user
        
        Args:
            user_id: User ID (1-indexed)
            n: Number of recommendations
            filter_already_rated: Whether to exclude already rated movies
            rating_matrix: Original rating matrix for filtering
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        # Get predictions for this user (returns array of shape (n_movies,))
        user_predictions = self.predict(user_id=user_id)
        
        logger.info(f"User predictions shape: {user_predictions.shape}")
        
        # Get movie IDs (1-indexed)
        n_movies = len(user_predictions)
        movie_ids = np.arange(1, n_movies + 1)
        
        # Filter already rated movies if requested
        if filter_already_rated and rating_matrix is not None:
            user_idx = user_id - 1
            # Get rated movies for this user (shape: (n_movies,))
            rated_movies = rating_matrix[user_idx].toarray().flatten()
            rated_mask = rated_movies > 0
            
            logger.info(f"Rated mask shape: {rated_mask.shape}")
            logger.info(f"User predictions shape: {user_predictions.shape}")
            
            # Make a copy and mask rated items
            user_predictions = user_predictions.copy()
            user_predictions[rated_mask] = -np.inf
        
        # Get top N
        top_indices = np.argsort(user_predictions)[::-1][:n]
        top_movie_ids = movie_ids[top_indices]
        top_ratings = user_predictions[top_indices]
        
        recommendations = list(zip(top_movie_ids, top_ratings))
        
        return recommendations

    
    def save_model(self, filepath):
        """Save trained model"""
        import pickle
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        model_data = {
            'config': self.config,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'global_mean': self.global_mean,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_biases = model_data.get('user_biases')
        self.item_biases = model_data.get('item_biases')
        self.global_mean = model_data.get('global_mean', 0.0)
        self.trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return self
