"""
Build Log-Likelihood-based term weights for movies
"""
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import EXTERNAL_DATA_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.features.movie_features import LogLikelihoodCalculator, MovieContentFeatures

logger = setup_logger("build_ll", LOGGING_CONFIG)

def main():
    logger.info("=" * 70)
    logger.info("BUILDING LOG-LIKELIHOOD TERM FEATURES")
    logger.info("=" * 70)

    plots_path = EXTERNAL_DATA_DIR / "movie_plots_by_id.json"
    if not plots_path.exists():
        logger.error(f"Plots file not found: {plots_path}")
        return

    with open(plots_path, "r", encoding="utf-8") as f:
        plots_dict = json.load(f)
        # Keys are strings; convert to int keys
        plots_dict = {int(k): v for k, v in plots_dict.items()}

    logger.info(f"Loaded plots for {len(plots_dict)} movies")

    # Step 1: Fit LL calculator
    ll_calc = LogLikelihoodCalculator(min_freq=3)
    ll_calc.fit(plots_dict)

    # Step 2: Build per-movie term weights
    feat_builder = MovieContentFeatures(ll_calc)
    movie_term_weights = feat_builder.build_movie_term_weights(plots_dict)

    # Save LL scores and movie term weights
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ll_path = EXTERNAL_DATA_DIR / "term_ll_scores.json"
    with open(ll_path, "w", encoding="utf-8") as f:
        json.dump(ll_calc.term_ll, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved term LL scores to {ll_path}")

    mtw_path = EXTERNAL_DATA_DIR / "movie_term_weights.json"
    with open(mtw_path, "w", encoding="utf-8") as f:
        # Convert int keys to str for JSON
        json.dump({str(k): v for k, v in movie_term_weights.items()}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved movie term weights to {mtw_path}")

    logger.info("=" * 70)
    logger.info("DONE BUILDING LL FEATURES")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
