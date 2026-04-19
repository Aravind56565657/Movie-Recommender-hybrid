"""
User similarity calculations for demographic features
Implements formulas 16, 17, 18 from the paper
"""
import numpy as np
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)

class UserSimilarityCalculator:
    """Calculate similarity between users based on demographics"""
    
    def __init__(self, config):
        """
        Args:
            config: Dict with age_groups, weights (L1, L2, L3)
        """
        self.config = config
        self.age_groups = config['age_groups']
        
    def age_similarity(self, age_group_a, age_group_b):
        """
        Calculate age similarity (Formula 16 from paper)
        
        Args:
            age_group_a, age_group_b: Age groups (1-7)
            
        Returns:
            Similarity score
        """
        diff = abs(age_group_a - age_group_b)
        
        if diff <= 1:
            return 1.0
        else:
            return 1.0 / diff
    
    def gender_similarity(self, gender_a, gender_b):
        """
        Calculate gender similarity (Formula 17 from paper)
        
        Args:
            gender_a, gender_b: 'M' or 'F'
            
        Returns:
            1 if same gender, 0 otherwise
        """
        return 1.0 if gender_a == gender_b else 0.0
    
    def genre_preference_similarity(self, genres_a, genres_b):
        """
        Calculate genre preference similarity using Jaccard (Formula 18)
        
        Args:
            genres_a, genres_b: Sets of preferred genres
            
        Returns:
            Jaccard similarity score
        """
        if not genres_a or not genres_b:
            return 0.0
        
        intersection = len(genres_a & genres_b)
        union = len(genres_a | genres_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def compute_similarity(self, user_a, user_b):
        """
        Compute overall user similarity (Formula 19 from paper)
        
        Args:
            user_a, user_b: User records with age_group, gender, preferred_genres
            
        Returns:
            Combined similarity score
        """
        # Individual similarities
        sim_age = self.age_similarity(
            user_a['age_group'], 
            user_b['age_group']
        )
        
        sim_gender = self.gender_similarity(
            user_a['gender'], 
            user_b['gender']
        )
        
        sim_genre = self.genre_preference_similarity(
            user_a['preferred_genres'], 
            user_b['preferred_genres']
        )
        
        # Weighted combination (L1, L2, L3 from paper)
        similarity = (
            self.config['weight_age'] * sim_age +
            self.config['weight_gender'] * sim_gender +
            self.config['weight_genre'] * sim_genre
        )
        
        return similarity
    
    def compute_similarity_matrix(self, users_df):
        """
        Compute pairwise similarity matrix for all users using memory-optimized operations.
        
        Args:
            users_df: DataFrame with user demographics
            
        Returns:
            Similarity matrix (n_users x n_users)
        """
        logger.info("Computing user similarity matrix (memory-optimized)...")
        
        n_users = len(users_df)
        
        # Initialize final matrix to accumulate similarities
        similarity_matrix = np.zeros((n_users, n_users), dtype=np.float32)
        
        # 1. Age Similarity
        age_groups = users_df['age_group'].values
        age_diffs = np.abs(age_groups[:, np.newaxis] - age_groups).astype(np.float32)
        mask_gt1 = age_diffs > 1
        age_diffs[mask_gt1] = 1.0 / age_diffs[mask_gt1]
        age_diffs[~mask_gt1] = 1.0
        
        similarity_matrix += self.config['weight_age'] * age_diffs
        del age_diffs, mask_gt1  # Free memory
        
        # 2. Gender Similarity
        genders = users_df['gender'].values
        sim_gender = (genders[:, np.newaxis] == genders).astype(np.float32)
        
        similarity_matrix += self.config['weight_gender'] * sim_gender
        del sim_gender
        
        # 3. Genre Preference Similarity (Jaccard)
        all_genres = set()
        for genres in users_df['preferred_genres']:
            all_genres.update(genres)
        all_genres = sorted(list(all_genres))
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}
        n_genres = len(all_genres)
        
        user_genre_matrix = np.zeros((n_users, n_genres), dtype=np.float32)
        for i, genres in enumerate(users_df['preferred_genres']):
            for genre in genres:
                if genre in genre_to_idx:
                    user_genre_matrix[i, genre_to_idx[genre]] = 1.0
        
        intersection = user_genre_matrix.dot(user_genre_matrix.T)
        user_genre_counts = user_genre_matrix.sum(axis=1)
        union = user_genre_counts[:, np.newaxis] + user_genre_counts - intersection
        
        mask_nonzero = union > 0
        # Compute in-place on intersection matrix to save memory
        intersection[mask_nonzero] /= union[mask_nonzero]
        intersection[~mask_nonzero] = 0.0
        
        similarity_matrix += self.config['weight_genre'] * intersection
        del intersection, union, user_genre_matrix, mask_nonzero
        
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"Mean similarity: {similarity_matrix.mean():.4f}")
        logger.info(f"Std similarity: {similarity_matrix.std():.4f}")
        
        return similarity_matrix
