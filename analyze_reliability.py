import numpy as np
import pandas as pd
from src.data.loader import load_movielens_1m_data
from src.data.preprocessor import preprocess_data
from src.models.als_model import ALSRecommender
from src.models.hybrid_model import HybridRecommender
from src.utils.config import *
from sklearn.model_selection import train_test_split

def analyze_segments():
    print("Loading data for reliability analysis...")
    ratings, users, movies = load_movielens_1m_data(RAW_DATA_DIR_1M)
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42, stratify=ratings['user_id']
    )
    processed_data = preprocess_data(train_ratings, users, movies, DEMOGRAPHIC_CONFIG)
    
    # 1. Train ALS (The state of the art component)
    als_model = ALSRecommender(ALS_CONFIG)
    als_model.fit(processed_data['rating_matrix'])
    als_preds = als_model.predict()
    
    # 2. Hybrid Components (With Dynamic Weights + Shrinkage)
    from src.models.demographic_model import DemographicRecommender
    from src.models.content_model import ContentRecommender
    from src.similarity.user_similarity import UserSimilarityCalculator
    from src.similarity.movie_similarity import MovieSimilarityCalculator
    
    baseline_info = als_model.get_baseline_info()
    
    # Demographic
    demo_model = DemographicRecommender(DEMOGRAPHIC_CONFIG)
    sim_calc = UserSimilarityCalculator(DEMOGRAPHIC_CONFIG)
    demo_model.fit(processed_data, sim_calc)
    demo_preds = demo_model.predict(baseline_info=baseline_info, top_n=DEMOGRAPHIC_CONFIG.get('neighborhood_size', 20))
    
    # Content
    n_movies = processed_data['rating_matrix'].shape[1]
    top_k_movie = SIMILARITY_CONFIG.get('top_n_similar', 20)
    movie_sim_calc = MovieSimilarityCalculator(top_k=top_k_movie)
    movie_embeddings = movie_sim_calc.compute_genre_tfidf(processed_data['movies'], n_movies)
    movie_sim_indices = movie_sim_calc.build_similarity(movie_embeddings, n_movies)
    
    content_model = ContentRecommender(top_sim_movies=top_k_movie)
    content_model.fit(processed_data['rating_matrix'], movie_sim_indices)
    content_preds = content_model.predict(baseline_info=baseline_info, shrinkage=5.0)
    
    # Hybrid Aggregator
    hybrid_model = HybridRecommender(HYBRID_CONFIG)
    hybrid_model.fit(als_preds, demo_preds, content_preds, user_rating_counts=processed_data['user_rating_counts'])
    hybrid_preds = hybrid_model.predict_matrix()
    
    # 3. Analyze Segments in Test Set
    user_rating_counts = processed_data['user_rating_counts']
    
    results = []
    for _, row in test_ratings.iterrows():
        u = int(row['user_id'])
        m = int(row['movie_id'])
        actual = row['rating']
        
        count = user_rating_counts.get(u, 0)
        group = "Cold" if count < 20 else ("Expert" if count > 100 else "Regular")
        
        # Predictions (0-indexed)
        p_als = als_preds[u-1, m-1]
        p_demo = demo_preds[u-1, m-1]
        p_cont = content_preds[u-1, m-1]
        p_hybrid = hybrid_preds[u-1, m-1]
        
        results.append({
            'group': group,
            'err_als': (actual - p_als)**2,
            'err_demo': (actual - p_demo)**2,
            'err_cont': (actual - p_cont)**2,
            'err_hybrid': (actual - p_hybrid)**2
        })
    
    df = pd.DataFrame(results)
    summary = df.groupby('group').agg({
        'err_als': lambda x: np.sqrt(x.mean()),
        'err_demo': lambda x: np.sqrt(x.mean()),
        'err_cont': lambda x: np.sqrt(x.mean()),
        'err_hybrid': lambda x: np.sqrt(x.mean())
    })
    
    report = f"\nReliability Report: ALS vs Hybrid RMSE\n"
    report += "-" * 50 + "\n"
    report += summary.to_string() + "\n"
    report += "-" * 50 + "\n"
    report += "Cold: < 20 ratings | Regular: 20-100 | Expert: > 100\n"
    
    print(report)
    with open('reliability_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    analyze_segments()
