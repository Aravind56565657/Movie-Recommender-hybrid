"""
Fetch Wikipedia plots for MovieLens movies
"""
import sys
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, WIKI_PLOTS_PATH, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.features.text_processing import WikipediaFetcher

logger = setup_logger("fetch_wiki", LOGGING_CONFIG)

def main():
    logger.info("=" * 70)
    logger.info("FETCHING WIKIPEDIA PLOTS FOR MOVIES")
    logger.info("=" * 70)

    # Load processed data (we only need movies)
    with open(PROCESSED_DATA_DIR / "processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    movies = data["movies"]

    logger.info(f"Total movies: {len(movies)}")

    # Initialize fetcher with cache path
    fetcher = WikipediaFetcher(WIKI_PLOTS_PATH)

    # For first run, maybe limit to e.g. 200 movies to test
    plots = fetcher.fetch_for_movies(movies, max_movies=None)

    # Save as separate file too (mapping movie_id -> text)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXTERNAL_DATA_DIR / "movie_plots_by_id.json"

    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plots, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved movie plots to {out_path}")
    logger.info("=" * 70)
    logger.info("DONE FETCHING WIKIPEDIA PLOTS")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
