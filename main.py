"""
Main entry point for Hybrid Movie Recommender System
"""
import argparse
from pathlib import Path
import numpy as np

from src.utils.config import *
from src.utils.logger import setup_logger
from src.utils.warm_start import generate_warm_start

# Setup logger
logger = setup_logger('main', LOGGING_CONFIG)

def run_pipeline(mode='full', dataset='ml-100k'):
    """
    Run the complete recommendation pipeline
    
    Args:
        mode: 'data' | 'als' | 'demographic' | 'content' | 'full'
    """
    logger.info(f"Starting pipeline in {mode} mode...")
    
    # -----------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------
    if mode in ['data', 'als', 'demographic', 'content', 'full']:
        logger.info(f"Step 0: Loading {dataset} data...")
        from src.data.loader import load_movielens_data, load_movielens_1m_data
        from src.data.preprocessor import preprocess_data
        from sklearn.model_selection import train_test_split
        
        # Load raw data based on dataset
        if dataset == 'ml-1m':
            data_path = RAW_DATA_DIR_1M
            load_fn = load_movielens_1m_data
        else:
            data_path = RAW_DATA_DIR_100K
            load_fn = load_movielens_data
            
        ratings, users, movies = load_fn(data_path)
        logger.info(f"[OK] Raw {dataset} data loaded.")

        # -----------------------------------------------------
        # 2. PREPROCESS
        # -----------------------------------------------------
        logger.info("Step 0: Preprocessing data...")
        # Preprocess ALL data first to get full user/movie mappings
        processed_data_full = preprocess_data(ratings, users, movies, DEMOGRAPHIC_CONFIG)
        
        # Create Train/Test split (80/20) on the original ratings
        logger.info("Splitting data into Train (80%) and Test (20%)...")
        train_ratings, test_ratings = train_test_split(
            ratings, test_size=0.2, random_state=42, stratify=ratings['user_id']
        )
        logger.info(f"Train size: {len(train_ratings)}, Test size: {len(test_ratings)}")

        # Preprocess TRAINING data only for model training
        processed_data_train = preprocess_data(train_ratings, users, movies, DEMOGRAPHIC_CONFIG)
        
        # Add test data to processed_data_train for evaluation later
        processed_data_train['test_ratings'] = test_ratings
        
        logger.info("[OK] Data preprocessing complete")

        # Extract training matrix for models
        train_matrix = processed_data_train['rating_matrix']
        
        # Initialize variables for predictions
        als_predictions = None
        demo_predictions = None
        content_predictions = None
        baseline_info = None

        # -----------------------------------------------------
        # 3. TRAIN MODELS
        # -----------------------------------------------------
        if mode == 'als' or mode == 'full':
            logger.info("Step 1: Training ALS model...")
            from src.models.als_model import ALSRecommender
            
            als_model = ALSRecommender(ALS_CONFIG)
            
            # Init Warm Start Factors
            user_factors, item_factors = generate_warm_start(
                train_ratings, # Use raw training ratings for history lookup
                processed_data_full['movies'], # Use full movie data for mapping
                train_matrix.shape[0],
                train_matrix.shape[1],
                ALS_CONFIG['factors']
            )
            
            als_model.fit(train_matrix, 
                          init_user_factors=user_factors,
                          init_item_factors=item_factors)
            
            als_predictions = als_model.predict()
            baseline_info = als_model.get_baseline_info()
            logger.info("[OK] ALS training complete")
        
        if mode == 'demographic' or mode == 'full':
            logger.info("Step 2: Computing demographic similarities...")
            from src.models.demographic_model import DemographicRecommender
            from src.similarity.user_similarity import UserSimilarityCalculator
            
            demo_model = DemographicRecommender(DEMOGRAPHIC_CONFIG)
            sim_calculator = UserSimilarityCalculator(DEMOGRAPHIC_CONFIG)
            
            # Note: Demographic model needs full user/movie metadata (from processed_data_full or train)
            # But the matrix should be train_matrix. 
            # processed_data_train contains the train matrix and subset of user/movies.
            demo_model.fit(processed_data_train, sim_calculator)
            
            # Pass baseline to demographic model
            demo_predictions = demo_model.predict(baseline_info=baseline_info)
            logger.info("[OK] Demographic model complete")
        
        if mode == 'content' or mode == 'full':
            logger.info("Step 3: Computing content similarities...")
            from src.models.content_model import ContentRecommender
            from src.similarity.movie_similarity import MovieSimilarityCalculator
            
            # 1. Compute movie embeddings using TF-IDF on genres
            n_movies = processed_data_train['rating_matrix'].shape[1]
            
            # 2. Compute movie similarity
            movie_sim_calc = MovieSimilarityCalculator(top_k=20)
            
            # Generate TF-IDF embeddings from processed movie data
            movie_embeddings = movie_sim_calc.compute_genre_tfidf(
                processed_data_train['movies'], 
                n_movies
            )
            
            movie_sim_indices = movie_sim_calc.build_similarity(movie_embeddings, n_movies)
            
            # 3. Train content model
            top_n = SIMILARITY_CONFIG.get('top_n_similar', 20)
            content_model = ContentRecommender(top_sim_movies=top_n)
            content_model.fit(processed_data_train['rating_matrix'], movie_sim_indices)
            
            # Pass baseline and shrinkage to content model
            content_predictions = content_model.predict(baseline_info=baseline_info, shrinkage=5.0)
            logger.info("[OK] Content model complete")
        
        final_predictions = None
        if mode == 'full':
            logger.info("Step 4: Combining hybrid predictions...")
            from src.models.hybrid_model import HybridRecommender
            
            # Prepare dataset stats for Neural Weight Generator
            # Efficiently calculate stats from sparse matrix
            user_counts = {u: train_matrix[u-1].nnz for u in range(1, train_matrix.shape[0] + 1)}
            movie_counts = {m: train_matrix[:, m-1].nnz for m in range(1, train_matrix.shape[1] + 1)}
            
            user_avgs = {}
            for u in range(1, train_matrix.shape[0] + 1):
                row = train_matrix[u-1]
                user_avgs[u] = row.data.mean() if row.nnz > 0 else 3.5
                
            movie_avgs = {}
            for m in range(1, train_matrix.shape[1] + 1):
                col = train_matrix[:, m-1]
                movie_avgs[m] = col.data.mean() if col.nnz > 0 else 3.5

            dataset_stats = {
                'user_counts': user_counts,
                'movie_counts': movie_counts,
                'user_avgs': user_avgs,
                'movie_avgs': movie_avgs,
                'max_u_count': 500,
                'max_i_count': 1000
            }
            
            hybrid_model = HybridRecommender(HYBRID_CONFIG)
            hybrid_model.fit(
                als_predictions,
                demo_predictions,
                content_predictions,
                user_rating_counts=processed_data_train.get('user_rating_counts'),
                dataset_stats=dataset_stats
            )
            
            # --- NEW: Train the Weight Generator ---
            if HYBRID_CONFIG.get('nn_weighting', False):
                hybrid_model.train_weight_generator(
                    train_ratings, 
                    dataset_stats, 
                    epochs=HYBRID_CONFIG.get('nn_epochs', 5), 
                    lr=HYBRID_CONFIG.get('nn_lr', 0.01)
                )
                
                # RE-FIT to generate new predictions using the TRAINED neural network
                hybrid_model.fit(
                    als_predictions,
                    demo_predictions,
                    content_predictions,
                    user_rating_counts=processed_data_train.get('user_rating_counts'),
                    dataset_stats=dataset_stats
                )
            
            final_predictions = hybrid_model.predict_matrix()
            logger.info("[OK] Hybrid model complete")
        elif mode == 'als':
            final_predictions = als_predictions
        elif mode == 'demographic':
            final_predictions = demo_predictions
        elif mode == 'content':
            final_predictions = content_predictions

        if final_predictions is not None:
            # Evaluate
            logger.info(f"Step 5: Evaluating model ({mode}) on Test Set...")
            from src.evaluation.evaluator import evaluate_model
            from src.utils.config import EVALUATION_CONFIG
            
            # We pass training matrix to mask known items during top-K recommendation
            train_matrix = processed_data_train.get('rating_matrix')
            test_ratings = processed_data_train.get('test_ratings')
            
            top_k_list = EVALUATION_CONFIG.get('top_k', [20])
            threshold = EVALUATION_CONFIG.get('relevance_threshold', 4.0)
            
            # This returns a results dict and also logs internally
            results = evaluate_model(final_predictions, test_ratings, train_matrix, k_list=top_k_list, threshold=threshold)
            
            logger.info("-" * 40)
            logger.info(f"FINAL RESULTS ({dataset} - {mode})")
            logger.info("-" * 40)
            logger.info(f"RMSE: {results['rmse']:.4f}")
            for k in top_k_list:
                logger.info(f"K={k} (Global):   P={results[f'p_global@{k}']:.4f}, R={results[f'r_global@{k}']:.4f}, F1={results[f'f1_global@{k}']:.4f}")
                logger.info(f"K={k} (TestOnly): P={results[f'p_test@{k}']:.4f}, R={results[f'r_test@{k}']:.4f}, F1={results[f'f1_test@{k}']:.4f}")
            logger.info("-" * 40)
            logger.info(f"HYBRID PIPELINE COMPLETE ({dataset})")
            logger.info("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid Movie Recommender')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['data', 'als', 'demographic', 'content', 'full'],
                       help='Pipeline mode to run')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m'],
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    # Update configuration based on dataset
    if args.dataset == 'ml-1m':
        DATASET_CONFIG['name'] = 'MovieLens-1M'
        DATASET_CONFIG['n_users'] = 6040
        DATASET_CONFIG['n_movies'] = 3952
        DATASET_CONFIG['n_ratings'] = 1000209
    
    run_pipeline(args.mode, args.dataset)
