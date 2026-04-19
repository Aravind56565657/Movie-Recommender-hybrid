"""
Test hybrid model: combine ALS, demographic, and content predictions
"""
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ALS_CONFIG,
    DEMOGRAPHIC_CONFIG,
    HYBRID_CONFIG,
    LOGGING_CONFIG,
)
from src.utils.logger import setup_logger
from src.models.als_model import ALSRecommender
from src.similarity.user_similarity import UserSimilarityCalculator
from src.models.demographic_model import DemographicRecommender
from src.similarity.movie_similarity import MovieSimilarityCalculator
from src.models.content_model import ContentRecommender
from src.models.hybrid_model import HybridRecommender
from src.evaluation.metrics import rmse

logger = setup_logger("test_hybrid", LOGGING_CONFIG)

def main():
    logger.info("=" * 70)
    logger.info("TESTING HYBRID MODEL")
    logger.info("=" * 70)

    # Load processed data
    with open(PROCESSED_DATA_DIR / "processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    rating_matrix = data["rating_matrix"]
    users = data["users"]
    movies = data["movies"]

    n_users, n_movies = rating_matrix.shape

    # ---- Step 1: ALS ----
    logger.info("\nStep 1: ALS predictions...")
    als_model = ALSRecommender(ALS_CONFIG)
    als_model.fit(rating_matrix)
    als_pred = als_model.predict()  # (n_users, n_movies)
    als_rmse = rmse(als_pred, rating_matrix)
    logger.info(f"ALS RMSE (train): {als_rmse:.4f}")

    # ---- Step 2: Demographic ----
    logger.info("\nStep 2: Demographic predictions...")
    sim_calc = UserSimilarityCalculator(DEMOGRAPHIC_CONFIG)
    demo_model = DemographicRecommender(DEMOGRAPHIC_CONFIG)
    demo_model.fit(data, sim_calc)
    demo_pred = demo_model._predict_all(top_n=20)
    demo_rmse = rmse(demo_pred, rating_matrix)
    logger.info(f"Demographic RMSE (train): {demo_rmse:.4f}")

    # ---- Step 3: Content ----
    logger.info("\nStep 3: Content predictions...")
    from src.utils.config import EXTERNAL_DATA_DIR
    import pickle as pkl

    emb_path = EXTERNAL_DATA_DIR / "movie_embeddings.pkl"
    with open(emb_path, "rb") as f:
        movie_embeddings = pkl.load(f)

    sim_calc_m = MovieSimilarityCalculator(top_k=50)
    top_sim_indices = sim_calc_m.build_similarity(movie_embeddings, n_movies)

    content_model = ContentRecommender(top_sim_movies=20)
    content_model.fit(rating_matrix, top_sim_indices)
    content_pred = content_model._predict_all()
    content_rmse = rmse(content_pred, rating_matrix)
    logger.info(f"Content RMSE (train): {content_rmse:.4f}")

    # ---- Step 4: Hybrid ----
    logger.info("\nStep 4: Hybrid combination...")
    hybrid_model = HybridRecommender(HYBRID_CONFIG)
    hybrid_model.fit(als_pred, demo_pred, content_pred)
    hybrid_pred = hybrid_model.predict_matrix()
    hybrid_rmse = rmse(hybrid_pred, rating_matrix)
    logger.info(f"Hybrid RMSE (train): {hybrid_rmse:.4f}")

    # Sample recommendations
    test_user_id = 1
    recs = hybrid_model.recommend(test_user_id, n=10, rating_matrix=rating_matrix)
    logger.info(f"\nTop-10 hybrid recommendations for User {test_user_id}:")
    for rank, (mid, score) in enumerate(recs, 1):
        title = movies[movies["movie_id"] == mid]["title"].values[0]
        logger.info(f"  {rank}. {mid} - {title[:40]} - {score:.3f}")

    logger.info("\n" + "=" * 70)
    logger.info("HYBRID MODEL TEST COMPLETE!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
