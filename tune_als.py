import numpy as np
import pandas as pd
from src.data.loader import load_movielens_1m_data
from src.data.preprocessor import preprocess_data
from src.models.als_model import ALSRecommender
from src.utils.config import RAW_DATA_DIR_1M, DEMOGRAPHIC_CONFIG, ALS_CONFIG
from sklearn.model_selection import train_test_split
import itertools

def tune_als():
    print("Loading data for tuning...")
    ratings, users, movies = load_movielens_1m_data(RAW_DATA_DIR_1M)
    
    # Stratified split to ensure all users are in train
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42, stratify=ratings['user_id']
    )
    
    # Calculate user rating counts from training data
    user_counts = train_ratings.groupby('user_id').size().to_dict()
    
    # Preprocess just enough for ALS
    processed_data = preprocess_data(train_ratings, users, movies, DEMOGRAPHIC_CONFIG)
    
    # ---------------------------------------------------------
    # WARM START PREPARATION (Use fixed random seed for consistency)
    # ---------------------------------------------------------
    n_items = processed_data['rating_matrix'].shape[1]
    
    # We will regenerate init_item_factors inside the loop if factors change,
    # or just pre-generate for max factors and slice? 
    # Better to consistent generate inside.
    
    def get_warm_start(n_factors):
        all_genres = set()
        for genres in movies['genres']:
            all_genres.update(genres) # genres is list
        sorted_genres = sorted(list(all_genres))
        
        np.random.seed(42)
        genre_vectors = {g: np.random.normal(0, 0.1, n_factors) for g in sorted_genres}
        
        init_item = np.random.normal(0, 0.1, (n_items, n_factors))
        
        # Map movie_id to index (id - 1)
        for _, row in movies.iterrows():
            mid = row['movie_id']
            idx = mid - 1
            if idx >= n_items: continue
            
            vecs = [genre_vectors[g] for g in row['genres'] if g in genre_vectors]
            if vecs:
                init_item[idx] = np.mean(vecs, axis=0)
        return init_item

    # ---------------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------------
    param_grid = {
        'factors': [10, 50],
        'regularization': [0.1, 10.0]
    }
    
    best_rmse = float('inf')
    best_params = None
    
    results = []
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\nStarting Grid Search: {len(combinations)} combinations")
    print("-" * 60)
    print(f"{'Factors':<10} {'Reg':<10} {'Cold RMSE':<15} {'All RMSE':<15}")
    print("-" * 60)
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        # Update config
        config = ALS_CONFIG.copy()
        config.update(params)
        # Reduce iterations for speed during tuning
        config['iterations'] = 5 
        
        # Init Model
        model = ALSRecommender(config)
        
        # Warm Start
        init_item = get_warm_start(params['factors'])
        
        # Train
        # Suppress massive logging if possible?
        model.fit(processed_data['rating_matrix'], init_item_factors=init_item)
        
        # Predict
        preds = model.predict()
        
        # Evaluate (Vectorized)
        test_users = test_ratings['user_id'].values - 1
        test_movies = test_ratings['movie_id'].values - 1
        test_actuals = test_ratings['rating'].values
        
        # Get predictions for test pairs
        test_preds = preds[test_users, test_movies]
        
        errors = (test_actuals - test_preds)**2
        
        # Create mask for cold users
        # Map user_id (1-indexed) to count
        # user_counts is dict {uid: count}
        # We need an array aligned with test_users
        
        # Precompute cold user set
        cold_user_ids = {u for u, c in user_counts.items() if c < 20}
        
        # Vectorized check: is test_user+1 in cold_user_ids?
        # Faster: create an array of counts indexed by user_id
        max_user_id = max(user_counts.keys())
        counts_arr = np.zeros(max_user_id + 1, dtype=int)
        for u, c in user_counts.items():
            counts_arr[u] = c
            
        # Get counts for test users (test_users is 0-indexed, so +1)
        test_user_counts = counts_arr[test_users + 1]
        cold_mask = test_user_counts < 20
        
        rmse_cold = np.sqrt(errors[cold_mask].mean()) if cold_mask.any() else 0.0
        rmse_all = np.sqrt(errors.mean())
        
        print(f"{params['factors']:<10} {params['regularization']:<10} {rmse_cold:<15.4f} {rmse_all:<15.4f}")
        
        if rmse_cold < best_rmse:
            best_rmse = rmse_cold
            best_params = params
            
    print("-" * 60)
    print(f"Best Probable Params for Cold Start: {best_params}")
    print(f"Best Cold RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    tune_als()
