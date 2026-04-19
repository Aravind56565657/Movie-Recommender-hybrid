import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_warm_start(ratings, movies, n_users, n_items, n_factors):
    """
    Generate initial user and item factors based on genre metadata.
    
    Args:
        ratings: DataFrame yielding high rated items for users
        movies: DataFrame with 'movie_id' and 'genres' (list)
        n_users: Number of users in matrix
        n_items: Number of items in matrix
        n_factors: Number of latent factors
        
    Returns:
        init_user_factors: (n_users, n_factors)
        init_item_factors: (n_items, n_factors)
    """
    logger.info("Generating Warm-Start Vectors using Genre Metadata...")
    
    # 1. Unique genres
    all_genres = set()
    for genres in movies['genres']:
        all_genres.update(genres)
    sorted_genres = sorted(list(all_genres))
    
    # 2. Assign random vector to each genre
    np.random.seed(42)
    # Use small variance
    genre_vectors = {g: np.random.normal(0, 0.1, n_factors) for g in sorted_genres}
    
    # 3. Init Item Factors
    # Initialize random first to cover items with no genres/mappings
    init_item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    # Map movie_id to genre vector
    movie_genre_map = {}
    
    # Pre-calculate movie vectors
    # Assumes movies['movie_id'] corresponds to the matrix columns via index=id-1
    # Check alignment carefully. In Preprocessor, we assume id-1.
    
    for _, row in movies.iterrows():
        mid = row['movie_id']
        idx = mid - 1
        
        if idx >= n_items: continue
        
        vecs = [genre_vectors[g] for g in row['genres'] if g in genre_vectors]
        if vecs:
            avg_vec = np.mean(vecs, axis=0)
            movie_genre_map[mid] = avg_vec
            init_item_factors[idx] = avg_vec
            
    # 4. Init User Factors
    init_user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
    
    # Filter for liked movies (Threshold >= 4.0)
    liked_movies = ratings[ratings['rating'] >= 4.0]
    
    # Group by user
    grouped = liked_movies.groupby('user_id')['movie_id'].apply(list)
    
    count_warmed = 0
    for user_id, movie_ids in grouped.items():
        u_idx = user_id - 1
        if u_idx >= n_users: continue
        
        vectors = []
        for mid in movie_ids:
            if mid in movie_genre_map:
                vectors.append(movie_genre_map[mid])
        
        if vectors:
            init_user_factors[u_idx] = np.mean(vectors, axis=0)
            count_warmed += 1
            
    logger.info(f"✓ Warm-started {len(movie_genre_map)} items")
    logger.info(f"✓ Warm-started {count_warmed} users")
    
    return init_user_factors, init_item_factors
