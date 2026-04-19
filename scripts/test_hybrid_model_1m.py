"""
Test hybrid model on MovieLens 1M
"""
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import (
    PROCESSED_DATA_DIR,
    ALS_CONFIG,
    DEMOGRAPHIC_CONFIG,
    HYBRID_CONFIG,
    EXTERNAL_DATA_DIR,
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

logger = setup_logger("test_hybrid_1m", LOGGING_CONFIG)

def main():
    logger.info("=" * 70)
    logger.info("TESTING HYBRID MODEL (ML-1M)")
    logger.info("=" * 70)

    # ---- load processed 1M data ----
    with open(PROCESSED_DATA_DIR / "processed_data_1m.pkl", "rb") as f:
        data = pickle.load(f)

    rating_matrix = data["rating_matrix"]
    users = data["users"]
    movies = data["movies"]
    n_users, n_movies = rating_matrix.shape

    # ---- Step 1: ALS ----
    logger.info("\nStep 1: ALS predictions (1M)...")
    als_model = ALSRecommender(ALS_CONFIG)
    als_model.fit(rating_matrix)
    als_pred = als_model.predict()
    als_rmse = rmse(als_pred, rating_matrix)
    logger.info(f"ALS RMSE (train, 1M): {als_rmse:.4f}")

    # ---- Step 2: Demographic ----
    logger.info("\nStep 2: Demographic predictions (1M)...")
    sim_calc = UserSimilarityCalculator(DEMOGRAPHIC_CONFIG)
    demo_model = DemographicRecommender(DEMOGRAPHIC_CONFIG)
    demo_model.fit(data, sim_calc)
    # Only first 500 users for speed
    max_users = 500
    logger.info(f"\nStep 2: Demographic predictions (1M) for first {max_users} users...")
    demo_pred = np.zeros((max_users, n_movies), dtype=np.float32)
    for u in range(max_users):
        if (u + 1) % 100 == 0:
            logger.info(f"  Demographic: user {u+1}/{max_users}")
        demo_pred[u] = demo_model.predict(user_id=u+1, top_n=20)
    demo_rmse = rmse(demo_pred, rating_matrix[:max_users])
    logger.info(f"Demographic RMSE (train, 1M, first {max_users} users): {demo_rmse:.4f}")

    # ---- Step 3: Content ----
    logger.info("\nStep 3: Content predictions (1M)...")
    # reuse same external pipeline: you would need plots/LL/embeddings for 1M too;
    # for a quick test, we can reuse the 100K embeddings if IDs overlap poorly,
    # but ideally you'd rebuild them for 1M.
    emb_path = EXTERNAL_DATA_DIR / "movie_embeddings.pkl"
    with open(emb_path, "rb") as f:
        movie_embeddings = pickle.load(f)

    sim_calc_m = MovieSimilarityCalculator(top_k=50)
    top_sim_indices = sim_calc_m.build_similarity(movie_embeddings, n_movies)

    content_model = ContentRecommender(top_sim_movies=20)
    content_model.fit(rating_matrix, top_sim_indices)
    logger.info(f"\nStep 3: Content predictions (1M) for first {max_users} users...")
    content_pred = np.zeros((max_users, n_movies), dtype=np.float32)
    for u in range(max_users):
        if (u + 1) % 100 == 0:
            logger.info(f"  Content: user {u+1}/{max_users}")
        content_pred[u] = content_model.predict(user_id=u+1)
    content_rmse = rmse(content_pred, rating_matrix[:max_users])
    logger.info(f"Content RMSE (train, 1M, first {max_users} users): {content_rmse:.4f}")

    # ---- Step 4: Hybrid ----
    logger.info("\nStep 4: Hybrid combination (1M)...")
    hybrid_model = HybridRecommender(HYBRID_CONFIG)
    hybrid_model.fit(als_pred[:max_users], demo_pred, content_pred)
    hybrid_pred = hybrid_model.predict_matrix()
    hybrid_rmse = rmse(hybrid_pred, rating_matrix[:max_users])
    logger.info(f"Hybrid RMSE (train, 1M, first {max_users} users): {hybrid_rmse:.4f}")


    test_user_id = 1
    recs = hybrid_model.recommend(test_user_id, n=10, rating_matrix=rating_matrix)
    logger.info(f"\nTop-10 hybrid recommendations for User {test_user_id} (1M):")
    for rank, (mid, score) in enumerate(recs, 1):
        title = movies[movies["movie_id"] == mid]["title"].values[0]
        logger.info(f"  {rank}. {mid} - {title[:40]} - {score:.3f}")

    logger.info("\n" + "=" * 70)
    logger.info("HYBRID MODEL (ML-1M) TEST COMPLETE!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
