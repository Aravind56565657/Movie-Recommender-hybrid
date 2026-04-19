"""
Test the data loading and preprocessing pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_movielens_data
from src.data.preprocessor import preprocess_data
from src.utils.config import RAW_DATA_DIR, DEMOGRAPHIC_CONFIG, PROCESSED_DATA_DIR
from src.utils.logger import setup_logger
from src.utils.config import LOGGING_CONFIG

import pickle

# Setup logger
logger = setup_logger('test_pipeline', LOGGING_CONFIG)

def main():
    logger.info("="*70)
    logger.info("TESTING DATA PIPELINE")
    logger.info("="*70)
    
    # Step 1: Load raw data
    logger.info("\nStep 1: Loading raw data...")
    ratings, users, movies = load_movielens_data(RAW_DATA_DIR)
    
    # Step 2: Preprocess
    logger.info("\nStep 2: Preprocessing...")
    processed_data = preprocess_data(ratings, users, movies, DEMOGRAPHIC_CONFIG)
    
    # Step 3: Inspect results
    logger.info("\nStep 3: Inspecting results...")
    logger.info(f"Users shape: {processed_data['users'].shape}")
    logger.info(f"Movies shape: {processed_data['movies'].shape}")
    logger.info(f"Rating matrix shape: {processed_data['rating_matrix'].shape}")
    
    # Sample user with preferences
    sample_user = processed_data['users'].iloc[0]
    logger.info(f"\nSample user:")
    logger.info(f"  ID: {sample_user['user_id']}")
    logger.info(f"  Age: {sample_user['age']} (Group: {sample_user['age_group']})")
    logger.info(f"  Gender: {sample_user['gender']}")
    logger.info(f"  Preferred genres: {sample_user['preferred_genres']}")
    
    # Save processed data
    logger.info("\nStep 4: Saving processed data...")
    save_path = PROCESSED_DATA_DIR / 'processed_data.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    logger.info(f"Saved to {save_path}")
    
    logger.info("\n" + "="*70)
    logger.info("DATA PIPELINE TEST COMPLETE!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
