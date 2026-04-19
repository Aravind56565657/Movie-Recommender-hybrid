import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.als_model import ALSRecommender
from src.utils.config import ALS_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger('test_als_1m', LOGGING_CONFIG)

def main():
    logger.info("="*70)
    logger.info("TESTING ALS MODEL (ML-1M)")
    logger.info("="*70)

    with open(PROCESSED_DATA_DIR / 'processed_data_1m.pkl', 'rb') as f:
        data = pickle.load(f)

    rating_matrix = data['rating_matrix']
    users = data['users']
    movies = data['movies']

    logger.info(f"Rating matrix shape: {rating_matrix.shape}")
    logger.info(f"Users: {len(users)}, Movies: {len(movies)}")

    als_model = ALSRecommender(ALS_CONFIG)
    als_model.fit(rating_matrix)
    pred = als_model.predict()

    from src.evaluation.metrics import rmse
    rmse_val = rmse(pred, rating_matrix)
    logger.info(f"ALS RMSE (train, 1M): {rmse_val:.4f}")

if __name__ == "__main__":
    main()
