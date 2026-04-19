"""
Test demographic-based user similarity model
"""
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.demographic_model import DemographicRecommender
from src.similarity.user_similarity import UserSimilarityCalculator
from src.utils.config import DEMOGRAPHIC_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger('test_demographic', LOGGING_CONFIG)

def main():
    logger.info("="*70)
    logger.info("TESTING DEMOGRAPHIC MODEL")
    logger.info("="*70)
    
    # Load processed data
    logger.info("\nLoading processed data...")
    with open(PROCESSED_DATA_DIR / 'processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Initialize similarity calculator
    logger.info("\nInitializing similarity calculator...")
    sim_calc = UserSimilarityCalculator(DEMOGRAPHIC_CONFIG)
    
    # Test individual similarities
    logger.info("\nTest 1: Individual similarities")
    user1 = data['users'].iloc[0]
    user2 = data['users'].iloc[1]
    
    logger.info(f"User 1: Age group {user1['age_group']}, Gender {user1['gender']}")
    logger.info(f"  Preferred genres: {user1['preferred_genres']}")
    logger.info(f"User 2: Age group {user2['age_group']}, Gender {user2['gender']}")
    logger.info(f"  Preferred genres: {user2['preferred_genres']}")
    
    sim = sim_calc.compute_similarity(user1, user2)
    logger.info(f"Similarity: {sim:.4f}")
    
    # Train demographic model
    logger.info("\nTraining demographic model...")
    demo_model = DemographicRecommender(DEMOGRAPHIC_CONFIG)
    demo_model.fit(data, sim_calc)
    
    # Test predictions
    logger.info("\nTest 2: Predictions")
    test_user_id = 1
    test_movie_id = 1
    
    pred = demo_model.predict(user_id=test_user_id, movie_id=test_movie_id)
    actual = data['rating_matrix'][test_user_id-1, test_movie_id-1]
    
    logger.info(f"User {test_user_id}, Movie {test_movie_id}")
    logger.info(f"  Predicted: {pred:.3f}")
    logger.info(f"  Actual: {actual:.3f}")
    logger.info(f"  Error: {abs(pred - actual):.3f}")
    
    # Full prediction matrix (small sample)
    logger.info("\nTest 3: Prediction matrix (first 10 users)...")
    predictions = np.zeros((10, data['rating_matrix'].shape[1]))
    for i in range(10):
        predictions[i] = demo_model.predict(user_id=i+1, top_n=20)
    
    logger.info(f"Prediction matrix shape: {predictions.shape}")
    logger.info(f"Mean: {predictions.mean():.3f}")
    logger.info(f"Std: {predictions.std():.3f}")
    
    logger.info("\n" + "="*70)
    logger.info("DEMOGRAPHIC MODEL TEST COMPLETE!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
