"""
Test content-based model using embeddings
"""
import sys
from pathlib import Path
import pickle
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.similarity.movie_similarity import MovieSimilarityCalculator
from src.models.content_model import ContentRecommender

logger = setup_logger("test_content", LOGGING_CONFIG)

def main():
    logger.info("=" * 70)
    logger.info("TESTING CONTENT-BASED MODEL")
    logger.info("=" * 70)

    # Load processed data
    with open(PROCESSED_DATA_DIR / "processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    rating_matrix = data["rating_matrix"]
    movies = data["movies"]

    # Load embeddings
    emb_path = EXTERNAL_DATA_DIR / "movie_embeddings.pkl"
    with open(emb_path, "rb") as f:
        movie_embeddings = pickle.load(f)

    n_movies = rating_matrix.shape[1]
    logger.info(f"Movies: {n_movies}, embeddings: {len(movie_embeddings)}")

    # Build similarity
    sim_calc = MovieSimilarityCalculator(top_k=50)
    top_sim_indices = sim_calc.build_similarity(movie_embeddings, n_movies)

    # Train content model
    content_model = ContentRecommender(top_sim_movies=20)
    content_model.fit(rating_matrix, top_sim_indices)

    # Test single prediction
    test_user_id = 1
    test_movie_id = 1
    pred = content_model.predict(user_id=test_user_id, movie_id=test_movie_id)
    actual = rating_matrix[test_user_id - 1, test_movie_id - 1]

    logger.info("\nTest 1: Single prediction")
    logger.info(f"User {test_user_id}, Movie {test_movie_id}")
    logger.info(f"  Predicted: {pred:.3f}")
    logger.info(f"  Actual: {actual:.3f}")

    # Small matrix for first 10 users
    logger.info("\nTest 2: Prediction matrix (first 5 users)...")
    preds = np.zeros((5, n_movies), dtype=np.float32)
    for u in range(5):
        preds[u] = content_model.predict(user_id=u + 1)

    logger.info(f"Preds shape: {preds.shape}")
    logger.info(f"Mean: {preds.mean():.3f}, Std: {preds.std():.3f}")

    logger.info("\n" + "=" * 70)
    logger.info("CONTENT-BASED MODEL TEST COMPLETE!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
