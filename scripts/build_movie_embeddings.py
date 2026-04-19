"""
Build dense movie embeddings from LL-weighted terms + Word2Vec
"""
import sys
from pathlib import Path
import json
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import EXTERNAL_DATA_DIR, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.features.movie_features import MovieEmbeddingBuilder

logger = setup_logger("build_emb", LOGGING_CONFIG)

def main():
    logger.info("=" * 70)
    logger.info("BUILDING MOVIE EMBEDDINGS (LL + Word2Vec)")
    logger.info("=" * 70)

    plots_path = EXTERNAL_DATA_DIR / "movie_plots_by_id.json"
    weights_path = EXTERNAL_DATA_DIR / "movie_term_weights.json"

    if not plots_path.exists() or not weights_path.exists():
        logger.error("Required files not found. Run fetch_wikipedia_plots.py and build_ll_features.py first.")
        return

    # Load plots
    with open(plots_path, "r", encoding="utf-8") as f:
        plots_dict = json.load(f)
        plots_dict = {int(k): v for k, v in plots_dict.items()}

    # Load term weights
    with open(weights_path, "r", encoding="utf-8") as f:
        movie_term_weights = json.load(f)
        movie_term_weights = {int(k): v for k, v in movie_term_weights.items()}

    logger.info(f"Movies with plots: {len(plots_dict)}")
    logger.info(f"Movies with term weights: {len(movie_term_weights)}")

    # Train Word2Vec
    emb_builder = MovieEmbeddingBuilder(
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
    )
    emb_builder.train_word2vec(plots_dict)

    # Build embeddings
    movie_embeddings = emb_builder.build_movie_embeddings(
        plots_dict, movie_term_weights
    )

    # Save embeddings and model
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = EXTERNAL_DATA_DIR / "movie_embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump(movie_embeddings, f)
    logger.info(f"Saved movie embeddings to {emb_path}")

    w2v_path = EXTERNAL_DATA_DIR / "word2vec_model.bin"
    emb_builder.model.save(str(w2v_path))
    logger.info(f"Saved Word2Vec model to {w2v_path}")

    logger.info("=" * 70)
    logger.info("DONE BUILDING MOVIE EMBEDDINGS")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
