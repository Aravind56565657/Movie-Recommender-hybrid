"""
Test ALS model training and prediction
"""
import sys
from pathlib import Path
import pickle
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.als_model import ALSRecommender
from src.utils.config import ALS_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger


# Setup logger
logger = setup_logger('test_als', LOGGING_CONFIG)

def main():
    logger.info("="*70)
    logger.info("TESTING ALS MODEL")
    logger.info("="*70)
    
    # Load processed data
    logger.info("\nLoading processed data...")
    with open(PROCESSED_DATA_DIR / 'processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    rating_matrix = data['rating_matrix']
    users = data['users']
    movies = data['movies']
    
    logger.info(f"Rating matrix shape: {rating_matrix.shape}")
    logger.info(f"Users: {len(users)}")
    logger.info(f"Movies: {len(movies)}")
    
    # Initialize and train ALS model
    logger.info("\nInitializing ALS model...")
    als_model = ALSRecommender(ALS_CONFIG)
    
    logger.info("\nTraining ALS model...")
    als_model.fit(rating_matrix)
    
    # Test predictions
    logger.info("\n" + "="*70)
    logger.info("TESTING PREDICTIONS")
    logger.info("="*70)
    
    # Test 1: Predict single rating
    test_user_id = 1
    test_movie_id = 1
    predicted_rating = als_model.predict(
        user_id=test_user_id, 
        movie_id=test_movie_id
    )
    
    # Get actual rating if exists
    actual_rating = rating_matrix[test_user_id-1, test_movie_id-1]
    
    logger.info(f"\nTest 1: Single prediction")
    logger.info(f"  User {test_user_id}, Movie {test_movie_id}")
    logger.info(f"  Predicted: {predicted_rating:.3f}")
    logger.info(f"  Actual: {actual_rating:.3f}")
    logger.info(f"  Error: {abs(predicted_rating - actual_rating):.3f}")
    
    # Test 2: Recommend movies for a user
    logger.info(f"\nTest 2: Top-10 recommendations for User {test_user_id}")
    recommendations = als_model.recommend(
        user_id=test_user_id,
        n=10,
        filter_already_rated=True,
        rating_matrix=rating_matrix
    )
    
    logger.info(f"\nTop 10 recommendations:")
    for rank, (movie_id, pred_rating) in enumerate(recommendations, 1):
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        logger.info(f"  {rank}. Movie {movie_id}: {movie_title[:40]} - Score: {pred_rating:.3f}")
    
    # Test 3: Get latent factors
    logger.info(f"\nTest 3: Latent factors")
    user_factors = als_model.get_user_factors()
    item_factors = als_model.get_item_factors()
    
    logger.info(f"  User factors shape: {user_factors.shape}")
    logger.info(f"  Item factors shape: {item_factors.shape}")
    logger.info(f"  Sample user vector (User 1): {user_factors[0][:5]}")  # First 5 dims
    
    # Test 4: Compute full prediction matrix
    logger.info(f"\nTest 4: Full prediction matrix")
    prediction_matrix = als_model.predict()
    logger.info(f"  Prediction matrix shape: {prediction_matrix.shape}")
    logger.info(f"  Min predicted rating: {prediction_matrix.min():.3f}")
    logger.info(f"  Max predicted rating: {prediction_matrix.max():.3f}")
    logger.info(f"  Mean predicted rating: {prediction_matrix.mean():.3f}")
    
    # Calculate RMSE on training data (just for sanity check)
    logger.info(f"\nTest 5: Training RMSE")
    mask = rating_matrix.toarray() > 0
    actual_ratings = rating_matrix.toarray()[mask]
    predicted_ratings = prediction_matrix[mask]
    
    rmse = np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2))
    logger.info(f"  Training RMSE: {rmse:.4f}")
    
    # Save model
    logger.info(f"\nSaving model...")
    model_path = MODELS_DIR / 'als_model.pkl'
    als_model.save_model(model_path)
    
    logger.info("\n" + "="*70)
    logger.info("ALS MODEL TEST COMPLETE!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
