"""
Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess MovieLens data for modeling"""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with age_groups, genre_threshold, etc.
        """
        self.config = config
        
    def map_age_to_group(self, age):
        """
        Map age to age group (1-7) as per paper
        
        Args:
            age: User's age
            
        Returns:
            Age group number (1-7)
        """
        for group, (min_age, max_age) in self.config['age_groups'].items():
            if min_age <= age <= max_age:
                return group
        return 7  # Default to oldest group
    
    def process_users(self, users):
        """
        Process user demographics
        
        Args:
            users: Raw users DataFrame
            
        Returns:
            Processed users DataFrame with age_group
        """
        logger.info("Processing user demographics...")
        
        users_processed = users.copy()
        
        # Map ages to groups
        users_processed['age_group'] = users_processed['age'].apply(
            self.map_age_to_group
        )
        
        logger.info(f"✓ Processed {len(users_processed)} users")
        logger.info(f"  Age group distribution:")
        for group in sorted(users_processed['age_group'].unique()):
            count = (users_processed['age_group'] == group).sum()
            logger.info(f"    Group {group}: {count} users")
        
        return users_processed
    
    def extract_genre_preferences(self, ratings, movies):
        """
        Extract genre preferences for each user (ratings >= threshold)
        
        Args:
            ratings: Ratings DataFrame
            movies: Movies DataFrame with genres
            
        Returns:
            DataFrame with user_id and preferred_genres (set)
        """
        logger.info("Extracting user genre preferences...")
        
        threshold = self.config['genre_threshold']
        
        # Get high ratings
        high_ratings = ratings[ratings['rating'] >= threshold].copy()
        
        # Merge with movie genres
        high_ratings = high_ratings.merge(
            movies[['movie_id', 'genres']], 
            on='movie_id'
        )
        
        # Aggregate genres per user
        user_genres = high_ratings.groupby('user_id')['genres'].apply(
            lambda x: set([genre for genres_list in x for genre in genres_list])
        ).reset_index()
        
        user_genres.columns = ['user_id', 'preferred_genres']
        
        logger.info(f"✓ Extracted preferences for {len(user_genres)} users")
        
        return user_genres
    
    def create_rating_matrix(self, ratings, shape=None):
        """
        Create user-movie rating matrix
        
        Args:
            ratings: Ratings DataFrame
            shape: Tuple (n_users, n_movies) or None for auto-detect
            
        Returns:
            Sparse rating matrix (CSR format)
        """
        logger.info("Creating rating matrix...")
        
        if shape is None:
            n_users = ratings['user_id'].max()
            n_movies = ratings['movie_id'].max()
        else:
            n_users, n_movies = shape
        
        # Create sparse matrix (user_id and movie_id are 1-indexed)
        row = ratings['user_id'].values - 1
        col = ratings['movie_id'].values - 1
        data = ratings['rating'].values
        
        rating_matrix = csr_matrix(
            (data, (row, col)), 
            shape=(n_users, n_movies),
            dtype=np.float32
        )
        
        # Calculate sparsity
        sparsity = 1.0 - (rating_matrix.nnz / (n_users * n_movies))
        
        logger.info(f"✓ Created matrix of shape {rating_matrix.shape}")
        logger.info(f"  Non-zero entries: {rating_matrix.nnz:,}")
        logger.info(f"  Sparsity: {sparsity*100:.2f}%")
        
        return rating_matrix
    
    def process_all(self, ratings, users, movies):
        """
        Process all data
        
        Args:
            ratings, users, movies: Raw DataFrames
            
        Returns:
            dict with processed data
        """
        logger.info("="*60)
        logger.info("PREPROCESSING DATA")
        logger.info("="*60)
        
        # Process users
        users_processed = self.process_users(users)
        
        # Extract genre preferences
        genre_prefs = self.extract_genre_preferences(ratings, movies)
        
        # Merge genre preferences with users
        users_processed = users_processed.merge(
            genre_prefs, 
            on='user_id', 
            how='left'
        )
        
        # Fill missing genre prefs with empty set
        users_processed['preferred_genres'] = users_processed['preferred_genres'].apply(
            lambda x: x if isinstance(x, set) else set()
        )
        
        # Create rating matrix
        # Use full users/movies to determine shape so test users/items fit
        n_users = users['user_id'].max()
        n_movies = movies['movie_id'].max()
        
        rating_matrix = self.create_rating_matrix(
            ratings, 
            shape=(n_users, n_movies)
        )
        
        # Calculate user rating counts (for dynamic weights)
        logger.info("Calculating user rating counts...")
        user_rating_counts = ratings.groupby('user_id').size().to_dict()
        
        logger.info("="*60)
        logger.info("✓ PREPROCESSING COMPLETE")
        logger.info("="*60)
        
        return {
            'ratings': ratings,
            'users': users_processed,
            'movies': movies,
            'rating_matrix': rating_matrix,
            'user_rating_counts': user_rating_counts,
        }


def preprocess_data(ratings, users, movies, config):
    """
    Convenience function for preprocessing
    
    Args:
        ratings, users, movies: Raw DataFrames
        config: Configuration dict
        
    Returns:
        dict with processed data
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.process_all(ratings, users, movies)
