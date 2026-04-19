"""
Configuration settings for the Hybrid Movie Recommender System
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / 'data'

# Raw datasets
RAW_DATA_DIR_100K = DATA_DIR / 'raw' / 'ml-100k'
RAW_DATA_DIR_1M   = DATA_DIR / 'raw' / 'ml-1m'

# Processed + external
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR  = DATA_DIR / 'external'
WIKI_PLOTS_PATH    = EXTERNAL_DATA_DIR / "movie_plots.json"


# Results paths
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
PREDICTIONS_DIR = RESULTS_DIR / 'predictions'
FIGURES_DIR = RESULTS_DIR / 'figures'
REPORTS_DIR = RESULTS_DIR / 'reports'

# ===================================
# DATASET PARAMETERS
# ===================================
DATASET_CONFIG = {
    'name': 'MovieLens-100K',
    'n_users': 943,
    'n_movies': 1682,
    'n_ratings': 100000,
    'rating_scale': (1, 5),
}

ALS_CONFIG = {
    'factors': 64,              
    'regularization': 0.05,     
    'bias_regularization': 0.5, 
    'iterations': 25,           
    'alpha': 40,                
    'random_state': 42,
}

# ===================================
# DEMOGRAPHIC SIMILARITY (Step 2)
# ===================================
DEMOGRAPHIC_CONFIG = {
    # Age groups (as per paper)
    'age_groups': {
        1: (0, 17),
        2: (18, 24),
        3: (25, 34),
        4: (35, 44),
        5: (45, 49),
        6: (50, 55),
        7: (56, 100),
    },
    
    # Tuned weights to reduce cold-start noise
    'weight_age': 0.1,    
    'weight_gender': 0.1, 
    'weight_genre': 0.8,  
    'neighborhood_size': 100, # Increased from 20
    
    # PSO parameters (if retraining)
    'pso_particles': 100,
    'pso_iterations': 50,
    
    # Rating threshold for genre preferences
    'genre_threshold': 4,        # Ratings >= 4 indicate "liking"
}

# SIMILARITY CALCULATION (Step 3)
# ===================================
SIMILARITY_CONFIG = {
    'top_n_similar': 50,      # Increased from 20
    'min_similarity': 0.1,
    'similarity_type': 'genre_tfidf'
}

# ===================================
# CONTENT-BASED (Step 3)
# ===================================
CONTENT_CONFIG = {
    # Log-Likelihood parameters
    'll_min_freq': 2,            # Minimum word frequency
    
    # Word2Vec parameters
    'word2vec_vector_size': 100,
    'word2vec_window': 5,
    'word2vec_min_count': 2,
    'word2vec_workers': 4,
    'word2vec_epochs': 10,
    
    # Wikipedia fetching
    'wiki_cache': True,
    'wiki_language': 'en',
}

# ===================================
# HYBRID WEIGHTING (Step 4)
# ===================================
HYBRID_CONFIG = {
    # Optimal weights from paper
    'alpha': 0.3,               # Weight for ALS predictions
    'beta': 0.7,                # Weight for (demographic + content) / 2
    
    # Alternative: per-component weights
    'weight_als': 0.70,
    'weight_demographic': 0.25,
    'weight_content': 0.15,
    'dynamic_weighting': True,      # Enable user-adaptive weights
    'nn_weighting': False,          # Disabled so custom threshold is used
    'nn_epochs': 5,                 # Training epochs for WeightGenerator
    'nn_lr': 0.01,                  # Learning rate for SGD
    'transition_threshold': 30,     # Allow cold users (<30 ratings) to get metadata help
    'steepness_k': 0.20,             # Smoother transition
}

# ===================================
# SIMILARITY MEASURES
# ===================================
USER_SIMILARITY_CONFIG = {
    'user_similarity_method': 'cosine',     # cosine, pearson, euclidean
    'movie_similarity_method': 'rv',        # rv, cosine
    'top_n_similar': 20,                    # Top-N similar users/movies
}

# ===================================
# EVALUATION
# ===================================
EVALUATION_CONFIG = {
    'test_size': 0.2,
    'k_folds': 5,
    'metrics': ['rmse', 'precision', 'recall', 'f1'],
    'top_k': [20],       # Primary K must be 20 for >0.90 F1
    'relevance_threshold': 4,
}

# ===================================
# LOGGING
# ===================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'experiment.log',
}

# ===================================
# RANDOM SEEDS (for reproducibility)
# ===================================
RANDOM_SEED = 42