"""
Step 4: Hybrid recommender – combine ALS, demographic, and content predictions
"""
import numpy as np
import logging
from src.models.weight_generator import WeightGenerator

logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Hybrid model: weighted sum of three prediction sources
    """
    def __init__(self, config):
        """
        Args:
            config: dict with weights for als, demographic, content
        """
        self.config = config
        self.w_als = config['weight_als']
        self.w_demo = config['weight_demographic']
        self.w_content = config['weight_content']
        
        # Neural Weight Generator (Advanced Optimization)
        self.weight_gen = WeightGenerator()

        total = self.w_als + self.w_demo + self.w_content
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Hybrid weights sum to {total}, normalizing to 1.0")
            self.w_als /= total
            self.w_demo /= total
            self.w_content /= total

        self.pred_als = None
        self.pred_demo = None
        self.pred_content = None
        self.hybrid_pred = None

    def _calculate_weights(self, user_rating_counts, n_users):
        """
        Calculate dynamic weights for each user based on rating count
        
        Returns:
            np.array of shape (n_users, 3) representing [w_als, w_demo, w_content]
        """
        if not self.config.get('dynamic_weighting', False) or user_rating_counts is None:
            # Use static weights for all users
            weights = np.zeros((n_users, 3))
            weights[:, 0] = self.w_als
            weights[:, 1] = self.w_demo
            weights[:, 2] = self.w_content
            return weights

        threshold = self.config.get('transition_threshold', 50)
        k = self.config.get('steepness_k', 0.1)
        
        # Initialize weight matrix
        W = np.zeros((n_users, 3))
        
        for u_idx in range(n_users):
            user_id = u_idx + 1
            count = user_rating_counts.get(user_id, 0)
            
            # Sigmoid for ALS importance: 1 / (1 + exp(-k * (count - threshold)))
            als_confidence = 1.0 / (1.0 + np.exp(-k * (count - threshold)))
            
            # Interpolate between Configured Base and Max Trust
            # If Config says w_als=0.8, we start there and go up to 0.98
            min_als = self.w_als
            max_als = 0.99
            
            w_als = min_als + (max_als - min_als) * als_confidence
            
            # Split remaining weight based on config ratios
            remaining = 1.0 - w_als
            
            sum_meta = self.w_demo + self.w_content
            if sum_meta < 1e-9:
                sum_meta = 1.0 # prevent div/0
                
            w_demo = remaining * (self.w_demo / sum_meta)
            w_content = remaining * (self.w_content / sum_meta)
            
            W[u_idx, 0] = w_als
            W[u_idx, 1] = w_demo
            W[u_idx, 2] = w_content
            
        return W

    def fit(self, als_pred, demo_pred, content_pred, user_rating_counts=None, dataset_stats=None):
        """
        Combine prediction matrices using dynamic or static weights

        Args:
            als_pred, demo_pred, content_pred: np.array (n_users, n_movies)
            user_rating_counts: dict {user_id: count} for dynamic weighting
            dataset_stats: dict for Neural Weight Generator
        """
        logger.info("=" * 60)
        if self.config.get('nn_weighting', False):
            logger.info("STEP 4: BUILDING DYNAMIC HYBRID PREDICTIONS (Neural Network)")
        elif self.config.get('dynamic_weighting', False):
            logger.info("STEP 4: BUILDING DYNAMIC HYBRID PREDICTIONS (Sigmoid)")
        else:
            logger.info("STEP 4: BUILDING STATIC HYBRID PREDICTIONS")
        logger.info("=" * 60)

        self.pred_als = als_pred
        self.pred_demo = demo_pred
        self.pred_content = content_pred

        n_users, n_items = self.pred_als.shape
        
        # Determine weighting strategy
        if self.config.get('nn_weighting', False) and dataset_stats:
            # Neural Weighting (Instance-specific)
            # For simplicity in this implementation, we apply it during prediction 
            # or pre-calculate it for known users. 
            # Here we'll generate the weights matrix.
            Weights = np.zeros((n_users, 3))
            logger.info("Generating instance-specific weights via Neural Network...")
            # Note: For full matrix prediction, we use user-specific weights (simplified)
            for u_idx in range(n_users):
                # Using a representative movie for each user to get their typical weight
                # (In a real production system, this would be per u,i pair)
                Weights[u_idx] = self.weight_gen.generate_weights(u_idx + 1, 1, dataset_stats)
        else:
            Weights = self._calculate_weights(user_rating_counts, n_users)
        
        if self.config.get('nn_weighting', False) and dataset_stats:
            # Full 3-component weighted sum for Neural Hybrid
            w_als = Weights[:, 0:1]
            w_demo = Weights[:, 1:2]
            w_content = Weights[:, 2:3]
            
            self.hybrid_pred = (
                w_als * self.pred_als +
                w_demo * self.pred_demo +
                w_content * self.pred_content
            )
        else:
            # Memory-Efficient Legacy Sigmoid/Static logic
            w_als = Weights[:, 0:1]
            # Memory-Efficient Chunked Logic
            w_als = Weights[:, 0:1]
            w_meta = 1.0 - w_als
            
            self.hybrid_pred = np.empty_like(self.pred_demo)
            chunk_size = 200
            for start in range(0, n_users, chunk_size):
                end = min(start + chunk_size, n_users)
                
                wa = w_als[start:end]
                wm = w_meta[start:end]
                
                # Formula: temp = wa*ALS + wm*(0.5*Demo + 0.5*Content)
                temp = self.pred_demo[start:end] * 0.5
                temp += self.pred_content[start:end] * 0.5
                temp *= wm
                temp += self.pred_als[start:end] * wa
                
                self.hybrid_pred[start:end] = temp
        
        # ADDED SIGNAL: If both metadata models agree on a direction, boost it
        # (Only for non-NN mode or as a safety layer)
        if not self.config.get('nn_weighting', False):
            # Chunked implementation for safety layer
            chunk_size = 200
            for start in range(0, n_users, chunk_size):
                end = min(start + chunk_size, n_users)
                
                wa = w_als[start:end]
                is_cold = (wa < 0.5).astype(np.float32)
                
                meta_signal = (self.pred_demo[start:end] + self.pred_content[start:end]) * 0.5
                
                chunk_pred = self.hybrid_pred[start:end]
                chunk_pred = (1.0 - 0.2 * is_cold) * chunk_pred + (0.2 * is_cold) * meta_signal
                self.hybrid_pred[start:end] = np.clip(chunk_pred, 1.0, 5.0)
        else:
            self.hybrid_pred = np.clip(self.hybrid_pred, 1.0, 5.0)

        logger.info(f"Hybrid matrix built and clipped with shape {self.hybrid_pred.shape}")
        return self

    def train_weight_generator(self, train_df, dataset_stats, epochs=5, lr=0.01, sample_size=50000):
        """
        Train the Neural Weight Generator using actual ratings
        """
        logger.info(f"Training Weight Generator for {epochs} epochs...")
        
        # Sample training data for efficiency
        if len(train_df) > sample_size:
            train_sample = train_df.sample(sample_size, random_state=42)
        else:
            train_sample = train_df

        for epoch in range(epochs):
            total_loss = 0
            for _, row in train_sample.iterrows():
                u_id = int(row['user_id'])
                m_id = int(row['movie_id'])
                r_actual = float(row['rating'])
                
                u_idx, m_idx = u_id - 1, m_id - 1
                
                # Get base predictions for this sample
                p_als = self.pred_als[u_idx, m_idx]
                p_demo = self.pred_demo[u_idx, m_idx]
                p_cont = self.pred_content[u_idx, m_idx]
                P = np.array([p_als, p_demo, p_cont])
                
                # Forward Pass
                w = self.weight_gen.generate_weights(u_id, m_id, dataset_stats)
                
                # Combined Prediction
                r_pred = np.dot(w, P)
                
                # Calculate Gradient of Loss wrt weights
                # Loss = (r_actual - r_pred)^2
                # dL/dr_pred = -2 * (r_actual - r_pred)
                # dr_pred/dw = P
                # dL/dw = -2 * (r_actual - r_pred) * P
                error = r_actual - r_pred
                total_loss += error**2
                dL_dw = -2.0 * error * P
                
                # Backward Pass (Updates weights)
                self.weight_gen.backward(dL_dw, learning_rate=lr)

            avg_rmse = np.sqrt(total_loss / len(train_sample))
            logger.info(f"Epoch {epoch+1}/{epochs} - Weight Generator Training RMSE: {avg_rmse:.4f}")

    def predict_matrix(self):
        """Return full hybrid prediction matrix"""
        if self.hybrid_pred is None:
            raise ValueError("Hybrid model not built yet. Call fit() first.")
        return self.hybrid_pred

    def predict_single(self, user_id, movie_id):
        """Predict single rating (1-indexed IDs)"""
        if self.hybrid_pred is None:
            raise ValueError("Hybrid model not built yet.")
        return float(self.hybrid_pred[user_id - 1, movie_id - 1])

    def recommend(self, user_id, n=10, rating_matrix=None):
        """
        Recommend top-N movies for a user, optionally filtering already rated

        Args:
            user_id: 1-indexed
            n: top-N
            rating_matrix: CSR matrix of original ratings (for filtering)

        Returns:
            list of (movie_id, score)
        """
        if self.hybrid_pred is None:
            raise ValueError("Hybrid model not built yet.")

        u = user_id - 1
        user_scores = self.hybrid_pred[u].copy()
        n_movies = user_scores.shape[0]
        movie_ids = np.arange(1, n_movies + 1)

        # Filter already rated
        if rating_matrix is not None:
            rated = rating_matrix[u].toarray().flatten() > 0
            user_scores[rated] = -np.inf

        top_idx = np.argsort(user_scores)[::-1][:n]
        top_movie_ids = movie_ids[top_idx]
        top_scores = user_scores[top_idx]

        return list(zip(top_movie_ids, top_scores))
