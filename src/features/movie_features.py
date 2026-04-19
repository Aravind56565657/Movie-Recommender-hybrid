"""
Movie content feature extraction:
- Build corpus from plots
- Compute Log-Likelihood (LL) scores for terms
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import logging
from collections import Counter
from gensim.models import Word2Vec
import numpy as np


logger = logging.getLogger(__name__)


class LogLikelihoodCalculator:
    """
    Compute Log-Likelihood scores for terms across movie plots
    (Simplified corpus-level version)
    """

    def __init__(self, min_freq=2):
        """
        Args:
            min_freq: minimum frequency for a term to be considered
        """
        self.min_freq = min_freq
        self.term_ll = {}  # term -> LL score

    def _tokenize(self, text):
        text = text.lower()
        # Keep only letters and numbers
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        return tokens

    def fit(self, plots_dict):
        """
        Compute LL scores over the whole corpus

        Args:
            plots_dict: {movie_id: text}
        """
        logger.info("Computing Log-Likelihood scores...")

        # 1) Count term frequencies in the whole corpus
        total_counts = Counter()
        total_tokens = 0

        for mid, text in plots_dict.items():
            if not text:
                continue
            tokens = self._tokenize(text)
            total_counts.update(tokens)
            total_tokens += len(tokens)

        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Unique terms: {len(total_counts)}")

        # 2) Compute LL vs a uniform baseline (simplified)
        #    For each term, compare observed vs expected under uniform distribution
        vocab_size = len(total_counts)
        if vocab_size == 0:
            logger.warning("No terms found in plots_dict")
            return self

        avg_count = total_tokens / vocab_size

        ll_scores = {}
        for term, obs in total_counts.items():
            if obs < self.min_freq:
                continue

            exp = avg_count
            # LL = 2 * ( O * log(O/E) ), ignoring background term
            # add small epsilon to avoid log(0)
            if obs > 0 and exp > 0:
                ll = 2.0 * (obs * math.log((obs + 1e-9) / exp))
            else:
                ll = 0.0
            ll_scores[term] = max(ll, 0.0)

        # Normalize (optional): scale LL to [0,1] for stability
        if ll_scores:
            max_ll = max(ll_scores.values())
            for term in ll_scores:
                ll_scores[term] /= max_ll

        self.term_ll = ll_scores
        logger.info(f"Computed LL for {len(self.term_ll)} terms")
        return self

    def get_term_score(self, term):
        """Get LL score for a term (default 0.0 if not seen)"""
        return self.term_ll.get(term.lower(), 0.0)

class MovieContentFeatures:
    """
    Build per-movie term weight dictionaries using LL scores
    """

    def __init__(self, ll_calculator):
        self.ll_calculator = ll_calculator

    def _tokenize(self, text):
        return LogLikelihoodCalculator()._tokenize(text)

    def build_movie_term_weights(self, plots_dict):
        """
        For each movie, build {term: weight} using LL scores

        Args:
            plots_dict: {movie_id: text}

        Returns:
            dict: {movie_id: {term: weight}}
        """
        logger.info("Building movie term-weight dictionaries...")
        movie_terms = {}

        for mid, text in plots_dict.items():
            if not text:
                movie_terms[mid] = {}
                continue

            tokens = self._tokenize(text)
            term_counts = Counter(tokens)

            # Weight each term by LL score * term frequency (optional)
            weights = {}
            for term, freq in term_counts.items():
                ll = self.ll_calculator.get_term_score(term)
                if ll > 0:
                    weights[term] = ll * freq

            movie_terms[mid] = weights

        logger.info(f"Built term weights for {len(movie_terms)} movies")
        return movie_terms

class MovieEmbeddingBuilder:
    """
    Build dense movie vectors using:
    - Word2Vec embeddings
    - LL-based term weights
    """
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def train_word2vec(self, plots_dict):
        """
        Train Word2Vec (CBOW) model on all plots
        """
        logger.info("Training Word2Vec model on plots...")

        sentences = []
        for text in plots_dict.values():
            if not text:
                continue
            tokens = self._tokenize(text)
            if tokens:
                sentences.append(tokens)

        logger.info(f"Total sentences: {len(sentences)}")

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=0,          # 0 = CBOW
            epochs=self.epochs,
        )

        logger.info("Word2Vec training complete")
        return self.model

    def build_movie_embeddings(self, plots_dict, movie_term_weights):
        """
        For each movie, build a dense vector:
        v_movie = sum( ll(term) * freq(term) * v_word ) / sum(weights)

        Args:
            plots_dict: {movie_id: text}
            movie_term_weights: {movie_id: {term: weight}}

        Returns:
            dict: {movie_id: np.array(vector_size,)}
        """
        if self.model is None:
            raise ValueError("Word2Vec model not trained yet")

        logger.info("Building movie embedding vectors...")

        movie_embeddings = {}
        dim = self.vector_size

        for mid, weights in movie_term_weights.items():
            if not weights:
                # No informative terms; use zero vector
                movie_embeddings[mid] = np.zeros(dim, dtype=np.float32)
                continue

            vec_sum = np.zeros(dim, dtype=np.float32)
            weight_sum = 0.0

            for term, w in weights.items():
                if term in self.model.wv:
                    vec_sum += w * self.model.wv[term]
                    weight_sum += w

            if weight_sum > 0:
                movie_embeddings[mid] = vec_sum / weight_sum
            else:
                movie_embeddings[mid] = np.zeros(dim, dtype=np.float32)

        logger.info(f"Built embeddings for {len(movie_embeddings)} movies")
        return movie_embeddings
