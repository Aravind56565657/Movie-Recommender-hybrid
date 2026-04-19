import sys
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_movielens_1m_data
from src.data.preprocessor import preprocess_data
from src.utils.config import RAW_DATA_DIR_1M, DEMOGRAPHIC_CONFIG, PROCESSED_DATA_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger('test_pipeline_1m', LOGGING_CONFIG)

def main():
    logger.info("="*70)
    logger.info("TESTING DATA PIPELINE (ML-1M)")
    logger.info("="*70)

    # Step 1: Load raw data
    logger.info("\nStep 1: Loading raw data (ML-1M)...")
    ratings, users, movies = load_movielens_1m_data(RAW_DATA_DIR_1M)

    # Step 2: Preprocess
    logger.info("\nStep 2: Preprocessing...")
    processed_data = preprocess_data(ratings, users, movies, DEMOGRAPHIC_CONFIG)

    # Step 3: Save processed data
    save_path = PROCESSED_DATA_DIR / 'processed_data_1m.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    logger.info(f"Saved ML-1M processed data to {save_path}")

    logger.info("\n" + "="*70)
    logger.info("DATA PIPELINE (ML-1M) TEST COMPLETE!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
