"""
Text processing: fetch Wikipedia plots and basic cleaning
"""
import wikipedia
import re
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class WikipediaFetcher:
    """
    Fetch and cache Wikipedia summaries for movies
    """
    def __init__(self, cache_path):
        """
        Args:
            cache_path: Path to JSON file to store plots
        """
        self.cache_path = Path(cache_path)
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded Wikipedia cache with {len(self.cache)} entries")
            except Exception:
                logger.warning("Failed to load existing cache, starting fresh")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Wikipedia cache with {len(self.cache)} entries")

    def clean_title(self, title):
        """
        Clean MovieLens title to basic movie name (remove year etc.)
        e.g. 'Toy Story (1995)' -> 'Toy Story'
        """
        # Remove year in parentheses
        return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

    def fetch_plot(self, title):
        """
        Fetch plot/summary for a single movie title
        Uses cache when possible
        """
        if title in self.cache:
            return self.cache[title]

        try:
            # Wikipedia search using cleaned title
            search_results = wikipedia.search(title)
            if not search_results:
                logger.warning(f"No Wikipedia result for: {title}")
                self.cache[title] = ""
                return ""

            # Take the first result
            page_title = search_results[0]
            page = wikipedia.page(page_title, auto_suggest=False)
            text = page.content

            # Optionally, we could try to extract 'Plot' section; for now use full content
            cleaned = self._basic_clean(text)
            self.cache[title] = cleaned
            time.sleep(0.2)  # be polite to API

            return cleaned

        except Exception as e:
            logger.warning(f"Error fetching '{title}': {e}")
            self.cache[title] = ""
            return ""

    def _basic_clean(self, text):
        """
        Basic text cleaning: remove references, excessive whitespace, etc.
        """
        # Remove brackets like [1], [2]
        text = re.sub(r"\[\d+\]", "", text)
        # Keep only basic characters
        text = re.sub(r"[^a-zA-Z0-9\s.,;:'\"!?-]", " ", text)
        # Normalize spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fetch_for_movies(self, movies_df, title_col="title", max_movies=None):
        """
        Fetch plots for all movies in DataFrame

        Args:
            movies_df: DataFrame with movie titles
            title_col: Column name for movie title
            max_movies: limit for testing; None = all

        Returns:
            dict: {movie_id: text}
        """
        logger.info("Fetching Wikipedia plots for movies...")
        plots = {}
        count = 0

        for _, row in movies_df.iterrows():
            movie_id = int(row["movie_id"])
            raw_title = row[title_col]
            clean = self.clean_title(raw_title)

            text = self.fetch_plot(clean)
            plots[movie_id] = text

            count += 1
            if count % 50 == 0:
                logger.info(f"Fetched {count} plots...")
                self._save_cache()

            if max_movies is not None and count >= max_movies:
                break

        self._save_cache()
        logger.info(f"Finished fetching plots for {len(plots)} movies")
        return plots
