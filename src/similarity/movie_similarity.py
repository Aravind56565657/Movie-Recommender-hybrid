"""
Movie similarity using content embeddings
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class MovieSimilarityCalculator:
    """
    Compute movie-movie similarity from embedding vectors
    """
    def __init__(self, top_k=50):
        self.top_k = top_k
        self.sim_matrix = None      # full similarity (optional)
        self.top_sim_indices = None # for each movie, top-K similar movies

    def build_similarity(self, movie_embeddings, n_movies):
        """
        Args:
            movie_embeddings: dict {movie_id: vector}
            n_movies: total number of movies in rating matrix

        Returns:
            top_sim_indices: np.array (n_movies, top_k) with movie indices
        """
        logger.info("Building movie similarity matrix from embeddings...")

        dim = len(next(iter(movie_embeddings.values())))
        # Build matrix in movie_id order (1..n_movies)
        emb_matrix = np.zeros((n_movies, dim), dtype=np.float32)
        for mid in range(1, n_movies + 1):
            vec = movie_embeddings.get(mid, np.zeros(dim, dtype=np.float32))
            emb_matrix[mid - 1] = vec

        # Cosine similarity
        sim = cosine_similarity(emb_matrix)
        self.sim_matrix = sim

        # For each movie, take top_k most similar (excluding itself)
        n = n_movies
        top_k = min(self.top_k, n - 1)
        top_indices = np.zeros((n, top_k), dtype=int)

        for i in range(n):
            scores = sim[i].copy()
            scores[i] = -np.inf
            idx = np.argsort(scores)[::-1][:top_k]
            top_indices[i] = idx

        self.top_sim_indices = top_indices
        logger.info(f"Similarity matrix shape: {sim.shape}")
        logger.info(f"Top-K indices shape: {top_indices.shape}")
        return top_indices

    def compute_genre_tfidf(self, movies_df, n_movies):
        """
        Compute TF-IDF vectors from movie genres
        
        Args:
            movies_df: DataFrame with 'movie_id' and 'genres' columns
            n_movies: total number of movies
            
        Returns:
            dict {movie_id: vector}
        """
        logger.info("Computing TF-IDF embeddings from genres...")
        
        # Ensure we have all movies up to n_movies
        # Create a list of genre strings for each movie ID 1..n_movies
        corpus = []
        movie_ids = []
        
        # Create a map for quick lookup
        movie_genre_map = {}
        for _, row in movies_df.iterrows():
            mid = row['movie_id']
            genres = row['genres'] # list of strings
            if isinstance(genres, list):
                genre_str = " ".join(genres)
            else:
                genre_str = ""
            movie_genre_map[mid] = genre_str
            
        # Build corpus in order of movie_id (1 to n_movies)
        for mid in range(1, n_movies + 1):
            corpus.append(movie_genre_map.get(mid, ""))
            movie_ids.append(mid)
            
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(token_pattern=r'[a-zA-Z0-9\-]+')
        tfidf_matrix = tfidf.fit_transform(corpus)
        
        # Convert to dict
        # tfidf_matrix is sparse, convert to dense for now (dim is small ~20 genres)
        dense_matrix = tfidf_matrix.toarray()
        
        embeddings = {}
        for i, mid in enumerate(movie_ids):
            embeddings[mid] = dense_matrix[i]
            
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return embeddings
