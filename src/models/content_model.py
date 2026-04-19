"""
Step 3: Content-based recommender using movie embeddings
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ContentRecommender:
    """
    Content-based CF: for each (user, movie),
    predict rating from similar movies that user has rated
    """
    def __init__(self, top_sim_movies=20):
        self.top_sim_movies = top_sim_movies
        self.movie_sim_indices = None  # from MovieSimilarityCalculator
        self.rating_matrix = None
        self.n_users = None
        self.n_movies = None
        self.trained = False

    def fit(self, rating_matrix, movie_sim_indices):
        """
        Args:
            rating_matrix: CSR (users x movies)
            movie_sim_indices: np.array (n_movies, K) of similar movie indices
        """
        logger.info("=" * 60)
        logger.info("STEP 3: TRAINING CONTENT-BASED MODEL")
        logger.info("=" * 60)

        self.rating_matrix = rating_matrix
        self.movie_sim_indices = movie_sim_indices
        self.n_users, self.n_movies = rating_matrix.shape
        self.trained = True

        logger.info(f"Rating matrix shape: {self.rating_matrix.shape}")
        logger.info(f"Movie sim indices shape: {self.movie_sim_indices.shape}")

        logger.info("=" * 60)
        logger.info("CONTENT MODEL READY")
        logger.info("=" * 60)
        return self

    def predict(self, user_id=None, movie_id=None, baseline_info=None, shrinkage=5.0):
        if not self.trained:
            raise ValueError("Model not trained yet.")

        if user_id is not None and movie_id is not None:
            return self._predict_single(user_id, movie_id)
        elif user_id is not None:
            return self._predict_for_user(user_id)
        else:
            return self._predict_all(baseline_info, shrinkage)

    def _predict_single(self, user_id, movie_id):
        u = user_id - 1
        m = movie_id - 1

        # similar movie indices for movie m (0-based)
        sim_movies = self.movie_sim_indices[m][: self.top_sim_movies]

        user_ratings = self.rating_matrix[u].toarray().flatten()
        sims = []

        num = 0.0
        den = 0.0

        for sm in sim_movies:
            r = user_ratings[sm]
            if r > 0:
                # similarity score from sim matrix (optional) or assume 1
                # if we had full sim matrix, we could use it; here treat all top-K equal
                s = 1.0
                num += s * r
                den += s

        if den == 0:
            # fallback to user's mean or global mean
            user_nonzero = user_ratings[user_ratings > 0]
            if len(user_nonzero) > 0:
                return user_nonzero.mean()
            else:
                return 3.0

        return num / den

    def _predict_for_user(self, user_id):
        u = user_id - 1
        preds = np.zeros(self.n_movies, dtype=np.float32)
        for m in range(self.n_movies):
            preds[m] = self._predict_single(user_id, m + 1)
        return preds

    def _predict_all(self, baseline_info=None, shrinkage=5.0):
        """Vectorized prediction for all users with Bayesian Shrinkage"""
        logger.info("Computing full content-based prediction matrix (vectorized + shrinkage)...")
        
        preds = np.zeros((self.n_users, self.n_movies), dtype=np.float32)
        R = self.rating_matrix.toarray()
        
        # 1. Base prediction (fallback and foundaton)
        if baseline_info:
            global_mean = baseline_info['global_mean']
            item_biases = baseline_info['item_biases']
            base_matrix = global_mean + item_biases[np.newaxis, :]
        else:
            # Fallback to user means
            user_sums = R.sum(axis=1)
            user_counts = (R > 0).sum(axis=1)
            user_means = np.zeros(self.n_users, dtype=np.float32)
            nonzero_user = user_counts > 0
            user_means[nonzero_user] = user_sums[nonzero_user] / user_counts[nonzero_user]
            user_means[~nonzero_user] = 3.0
            base_matrix = user_means[:, np.newaxis]
            
        for m in range(self.n_movies):
            top_k_indices = int(self.top_sim_movies)
            sim_indices = self.movie_sim_indices[m][:top_k_indices].astype(np.int32)
            
            # Neighborhood information
            sim_ratings = R[:, sim_indices] # (n_users, k)
            mask = sim_ratings > 0
            
            # BIAS CORRECTION: Subtract item popularity (item_biases) from neighbor ratings
            # so we only look at how much this user LIKED similar items relative to others
            if baseline_info:
                sim_biases = item_biases[sim_indices]
                # subtract from ratings (broadcasting)
                sim_residuals = np.where(mask, sim_ratings - (global_mean + sim_biases), 0.0)
                sums = sim_residuals.sum(axis=1)
            else:
                sums = sim_ratings.sum(axis=1)
                
            counts = mask.sum(axis=1).astype(np.float32)
            
            # Bayesian Shrinkage toward the movie's own baseline popularity
            baseline_for_movie = base_matrix[:, m]
            
            # Prediction = Baseline + Smoothed Residual
            preds[:, m] = baseline_for_movie + (sums / (counts + shrinkage))
            
        logger.info(f"Prediction matrix shape: {preds.shape} (with shrinkage)")
        return preds
