"""
Data loading utilities for MovieLens 100K dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MovieLensLoader:
    """Load MovieLens 100K dataset"""
    
    def __init__(self, data_path):
        """
        Args:
            data_path: Path to ml-100k folder
        """
        self.data_path = Path(data_path)
        self.genre_names = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
    def load_ratings(self, filename='u.data'):
        """
        Load ratings data
        
        Returns:
            DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        logger.info(f"Loading ratings from {filename}...")
        
        ratings = pd.read_csv(
            self.data_path / filename,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        logger.info(f"✓ Loaded {len(ratings):,} ratings")
        logger.info(f"  Users: {ratings['user_id'].nunique()}")
        logger.info(f"  Movies: {ratings['movie_id'].nunique()}")
        
        return ratings
    
    def load_users(self, filename='u.user'):
        """
        Load user demographic data
        
        Returns:
            DataFrame with columns: user_id, age, gender, occupation, zip_code
        """
        logger.info(f"Loading users from {filename}...")
        
        users = pd.read_csv(
            self.data_path / filename,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        logger.info(f"✓ Loaded {len(users)} users")
        
        return users
    
    def load_movies(self, filename='u.item'):
        """
        Load movie information with genres
        
        Returns:
            DataFrame with columns: movie_id, title, release_date, ..., genres (list)
        """
        logger.info(f"Loading movies from {filename}...")
        
        # Column names
        movie_cols = ['movie_id', 'title', 'release_date', 
                      'video_release_date', 'imdb_url'] + self.genre_names
        
        movies = pd.read_csv(
            self.data_path / filename,
            sep='|',
            names=movie_cols,
            encoding='latin-1',
            engine='python'
        )
        
        # Create genres list column
        movies['genres'] = movies[self.genre_names].apply(
            lambda x: [self.genre_names[i] for i, val in enumerate(x) if val == 1],
            axis=1
        )
        
        logger.info(f"✓ Loaded {len(movies)} movies")
        
        return movies
    
    def load_train_test_split(self, fold=1):
        """
        Load pre-made train/test split
        
        Args:
            fold: Fold number (1-5) for u1-u5, or 'a'/'b' for ua/ub
            
        Returns:
            train_ratings, test_ratings DataFrames
        """
        logger.info(f"Loading train/test split for fold {fold}...")
        
        if isinstance(fold, int) and 1 <= fold <= 5:
            train_file = f'u{fold}.base'
            test_file = f'u{fold}.test'
        elif fold in ['a', 'b']:
            train_file = f'u{fold}.base'
            test_file = f'u{fold}.test'
        else:
            raise ValueError("Fold must be 1-5 or 'a'/'b'")
        
        train_ratings = self.load_ratings(train_file)
        test_ratings = self.load_ratings(test_file)
        
        logger.info(f"  Train: {len(train_ratings):,} ratings")
        logger.info(f"  Test: {len(test_ratings):,} ratings")
        
        return train_ratings, test_ratings
    
    def load_all(self):
        """
        Load all data files
        
        Returns:
            dict with 'ratings', 'users', 'movies'
        """
        return {
            'ratings': self.load_ratings(),
            'users': self.load_users(),
            'movies': self.load_movies(),
        }


def load_movielens_data(data_path):
    """
    Convenience function to load MovieLens data
    
    Args:
        data_path: Path to ml-100k folder
        
    Returns:
        ratings, users, movies DataFrames
    """
    loader = MovieLensLoader(data_path)
    data = loader.load_all()
    return data['ratings'], data['users'], data['movies']

class MovieLens1MLoader:
    """Load MovieLens 1M dataset"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load_ratings(self, filename='ratings.dat'):
        # userId::movieId::rating::timestamp
        ratings = pd.read_csv(
            self.data_path / filename,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        logger.info(f"ML-1M: {len(ratings):,} ratings, "
                    f"Users={ratings.user_id.nunique()}, "
                    f"Movies={ratings.movie_id.nunique()}")
        return ratings

    def load_users(self, filename='users.dat'):
        # userId::gender::age::occupation::zip
        users = pd.read_csv(
            self.data_path / filename,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
        )
        logger.info(f"ML-1M: {len(users)} users")
        return users

    def load_movies(self, filename='movies.dat'):
        # movieId::title::genres (pipe-separated genres)
        movies = pd.read_csv(
            self.data_path / filename,
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres_raw'],
            encoding='latin-1'
        )
        # convert "Action|Comedy" → list
        movies['genres'] = movies['genres_raw'].apply(
            lambda g: g.split('|') if isinstance(g, str) else []
        )
        logger.info(f"ML-1M: {len(movies)} movies")
        return movies

    def load_all(self):
        return {
            'ratings': self.load_ratings(),
            'users'  : self.load_users(),
            'movies' : self.load_movies(),
        }


def load_movielens_1m_data(data_path):
    loader = MovieLens1MLoader(data_path)
    data = loader.load_all()
    return data['ratings'], data['users'], data['movies']

