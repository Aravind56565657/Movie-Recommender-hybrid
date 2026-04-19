"""
Step 2: Demographic-based User Similarity Model
Uses age, gender, and genre preferences to compute user similarity
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class DemographicRecommender:
    """
    Demographic-based collaborative filtering
    Implements Step 2 from the paper
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with demographic weights
        """
        self.config = config
        self.similarity_matrix = None
        self.users_df = None
        self.rating_matrix = None
        self.trained = False
        
    def fit(self, data, similarity_calculator):
        """
        Compute user similarities based on demographics
        
        Args:
            data: Dict with 'users', 'rating_matrix'
            similarity_calculator: UserSimilarityCalculator instance
        """
        logger.info("="*60)
        logger.info("STEP 2: COMPUTING DEMOGRAPHIC SIMILARITIES")
        logger.info("="*60)
        
        self.users_df = data['users']
        self.rating_matrix = data['rating_matrix']
        
        # Compute similarity matrix
        self.similarity_matrix = similarity_calculator.compute_similarity_matrix(
            self.users_df
        )
        
        self.trained = True
        
        logger.info("="*60)
        logger.info("DEMOGRAPHIC MODEL COMPLETE")
        logger.info("="*60)
        
        return self
    
    def predict(self, user_id=None, movie_id=None, top_n=20, baseline_info=None):
        """
        Predict ratings using similar users (Formula 21 from paper)
        
        Args:
            user_id: User ID (1-indexed) or None for all
            movie_id: Movie ID (1-indexed) or None for all
            top_n: Number of similar users to consider
            baseline_info: dict with 'global_mean' and 'item_biases'
            
        Returns:
            Predicted rating(s)
        """
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        if user_id is not None:
            return self._predict_for_user(user_id, movie_id, top_n)
        else:
            return self._predict_all(top_n, baseline_info)
    
    def _predict_for_user(self, user_id, movie_id=None, top_n=20):
        """Predict ratings for a single user"""
        user_idx = user_id - 1
        
        # Get similar users
        similarities = self.similarity_matrix[user_idx].copy()
        similarities[user_idx] = 0  # Exclude self
        
        # Get top-N similar users
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_sims = similarities[top_indices]
        
        if top_sims.sum() == 0:
            # No similar users, return mean rating
            return np.mean(self.rating_matrix[user_idx].data) if self.rating_matrix[user_idx].nnz > 0 else 3.0
        
        # Get ratings from similar users
        similar_user_ratings = self.rating_matrix[top_indices].toarray()
        
        # User's mean rating
        user_mean = np.mean(self.rating_matrix[user_idx].data) if self.rating_matrix[user_idx].nnz > 0 else 3.0
        
        if movie_id is not None:
            # Single movie prediction
            movie_idx = movie_id - 1
            
            # Weighted average of deviations (Formula 21)
            numerator = 0
            denominator = 0
            
            for i, sim_user_idx in enumerate(top_indices):
                sim_rating = similar_user_ratings[i, movie_idx]
                if sim_rating > 0:  # User has rated this movie
                    sim_user_mean = np.mean(self.rating_matrix[sim_user_idx].data)
                    numerator += top_sims[i] * (sim_rating - sim_user_mean)
                    denominator += top_sims[i]
            
            if denominator == 0:
                return user_mean
            
            prediction = user_mean + (numerator / denominator)
            return prediction
        
        else:
            # Predict all movies
            n_movies = self.rating_matrix.shape[1]
            predictions = np.zeros(n_movies)
            
            for movie_idx in range(n_movies):
                numerator = 0
                denominator = 0
                
                for i, sim_user_idx in enumerate(top_indices):
                    sim_rating = similar_user_ratings[i, movie_idx]
                    if sim_rating > 0:
                        sim_user_mean = np.mean(self.rating_matrix[sim_user_idx].data)
                        numerator += top_sims[i] * (sim_rating - sim_user_mean)
                        denominator += top_sims[i]
                
                if denominator > 0:
                    predictions[movie_idx] = user_mean + (numerator / denominator)
                else:
                    predictions[movie_idx] = user_mean
            
            return predictions
    
    def _predict_all(self, top_n=20, baseline_info=None):
        """Predict full rating matrix using vectorized operations"""
        logger.info("Computing full demographic prediction matrix (vectorized)...")
        
        n_users = self.rating_matrix.shape[0]
        n_movies = self.rating_matrix.shape[1]
        
        # 1. Base prediction (fallback and foundation)
        if baseline_info:
            global_mean = baseline_info['global_mean']
            item_biases = baseline_info['item_biases']
            # Foundation is movie-specific bias
            base_matrix = global_mean + item_biases[np.newaxis, :]
        else:
            # Fallback to user mean if no ALS bias provided
            user_sums = np.array(self.rating_matrix.sum(axis=1)).flatten()
            user_counts = np.diff(self.rating_matrix.indptr)
            user_means = np.zeros(n_users)
            user_means[user_counts > 0] = user_sums[user_counts > 0] / user_counts[user_counts > 0]
            base_matrix = user_means[:, np.newaxis]
            
        # 2. Subtract means for centered rating logic
        # For simplicity and consistency with the paper, we still use user-mean centered ratings
        # to find how much neighbors deviate from THEIR norm.
        user_sums_raw = np.array(self.rating_matrix.sum(axis=1)).flatten()
        user_counts_raw = np.diff(self.rating_matrix.indptr)
        u_means = np.zeros(n_users)
        u_means[user_counts_raw > 0] = user_sums_raw[user_counts_raw > 0] / user_counts_raw[user_counts_raw > 0]
        u_means[user_counts_raw == 0] = 3.0 # Default mean for users with no ratings
        
        rating_matrix_centered = self.rating_matrix.copy().astype(np.float32)
        rows, cols = rating_matrix_centered.nonzero()
        rating_matrix_centered.data -= u_means[rows]
        
        # 3. For each user, find top-N similar users
        # We'll zero out self-similarity first
        np.fill_diagonal(self.similarity_matrix, 0)
        
        # Create a filtered similarity matrix keeping only top-N per row
        # This is faster than sorting everything if N << n_users
        sim_top_n = np.zeros_like(self.similarity_matrix)
        for i in range(n_users):
            row = self.similarity_matrix[i]
            # Get indices of top N
            top_indices = np.argpartition(row, -top_n)[-top_n:]
            # Keep only these
            sim_top_n[i, top_indices] = row[top_indices]
            
        # 4. Compute weighted sum of centered ratings
        # Prediction = Mean + (Sum(Sim * (Rating - Mean)) / Sum(|Sim|))
        # Part 2 is just dot product of Sim matrix and Centered Rating matrix
        
        # Convert top-N sim matrix to sparse for efficiency
        sim_top_n_sparse = csr_matrix(sim_top_n)
        
        # Weighted sum of deviations
        # (N_users x N_users) . (N_users x N_movies) -> (N_users x N_movies)
        pred_deviations = sim_top_n_sparse.dot(rating_matrix_centered)
        
        # Sum of weights (absolute similarities) for normalization
        # We only sum weights for users who actually rated the item
        # sum_weights = sim_top_n_sparse.dot((self.rating_matrix != 0).astype(float))
        # Note: (self.rating_matrix != 0) is the binary indicator matrix R_bin
        R_bin = self.rating_matrix.astype(bool).astype(np.float32)
        sum_weights = sim_top_n_sparse.dot(R_bin)
        
        # Avoid division by zero
        sum_weights_safe = sum_weights.toarray()
        sum_weights_safe[sum_weights_safe == 0] = 1.0
        
        # 5. Final prediction
        pred_deviations_dense = pred_deviations.toarray()
        weighted_avg_deviations = pred_deviations_dense / sum_weights_safe
        
        # Add base (could be user means or global_mean + item_biases)
        prediction_matrix = base_matrix + weighted_avg_deviations
        
        # Handle cases where no similar users rated the movie (sum_weights was 0)
        # In those cases, the weighted_avg_dev is 0, so we just return user_mean
        # This is already handled by the logic above.
        
        logger.info(f"Prediction matrix shape: {prediction_matrix.shape}")
        return prediction_matrix
