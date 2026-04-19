import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set path to your dataset
from src.utils.config import RAW_DATA_DIR_100K
DATA_PATH = RAW_DATA_DIR_100K

print("="*50)
print("PHASE 2: DATA LOADING & EXPLORATION")
print("="*50)

# ===================================
# 1. LOAD RATINGS DATA (u.data)
# ===================================
print("\n1. Loading ratings data (u.data)...")

ratings = pd.read_csv(
    f"{DATA_PATH}/u.data",
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    engine='python'
)

print(f"✓ Loaded {len(ratings):,} ratings")
print(f"  Users: {ratings['user_id'].nunique()}")
print(f"  Movies: {ratings['movie_id'].nunique()}")
print(f"  Sparsity: {(1 - len(ratings) / (ratings['user_id'].nunique() * ratings['movie_id'].nunique())) * 100:.2f}%")

print("\nFirst 5 ratings:")
print(ratings.head())

print("\nRating distribution:")
print(ratings['rating'].value_counts().sort_index())


# ===================================
# 2. LOAD USER DATA (u.user)
# ===================================
print("\n2. Loading user data (u.user)...")

users = pd.read_csv(
    f"{DATA_PATH}/u.user",
    sep='|',
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
    engine='python'
)

print(f"✓ Loaded {len(users)} users")
print("\nFirst 5 users:")
print(users.head())

# Map ages to groups (as per paper)
def age_to_group(age):
    if age < 18:
        return 1
    elif age <= 24:
        return 2
    elif age <= 34:
        return 3
    elif age <= 44:
        return 4
    elif age <= 49:
        return 5
    elif age <= 55:
        return 6
    else:
        return 7

users['age_group'] = users['age'].apply(age_to_group)

print("\nAge group distribution:")
print(users['age_group'].value_counts().sort_index())

print("\nGender distribution:")
print(users['gender'].value_counts())


# ===================================
# 3. LOAD MOVIE DATA (u.item)
# ===================================
print("\n3. Loading movie data (u.item)...")

# Genre names (from u.genre)
genre_names = [
    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

# Column names for u.item
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_names

movies = pd.read_csv(
    f"{DATA_PATH}/u.item",
    sep='|',
    names=movie_cols,
    encoding='latin-1',
    engine='python'
)

print(f"✓ Loaded {len(movies)} movies")
print("\nFirst 5 movies:")
print(movies[['movie_id', 'title'] + genre_names[:5]].head())

# Get genres for each movie (as list)
movies['genres'] = movies[genre_names].apply(
    lambda x: [genre_names[i] for i, val in enumerate(x) if val == 1], 
    axis=1
)

print("\nMost common genres:")
all_genres = [g for genres_list in movies['genres'] for g in genres_list]
from collections import Counter
print(Counter(all_genres).most_common(5))


# ===================================
# 4. CREATE RATING MATRIX
# ===================================
print("\n4. Creating sparse rating matrix...")

# Create user-movie matrix
rating_matrix = ratings.pivot(
    index='user_id',
    columns='movie_id',
    values='rating'
).fillna(0)  # Fill missing with 0 for now

print(f"✓ Rating matrix shape: {rating_matrix.shape}")
print(f"  Non-zero entries: {(rating_matrix != 0).sum().sum():,}")
print(f"  Total entries: {rating_matrix.size:,}")
print(f"  Sparsity: {((rating_matrix == 0).sum().sum() / rating_matrix.size) * 100:.2f}%")


# ===================================
# 5. BASIC STATISTICS
# ===================================
print("\n5. Basic statistics...")

# Ratings per user
ratings_per_user = ratings.groupby('user_id').size()
print(f"\nRatings per user:")
print(f"  Mean: {ratings_per_user.mean():.1f}")
print(f"  Median: {ratings_per_user.median():.1f}")
print(f"  Min: {ratings_per_user.min()}")
print(f"  Max: {ratings_per_user.max()}")

# Ratings per movie
ratings_per_movie = ratings.groupby('movie_id').size()
print(f"\nRatings per movie:")
print(f"  Mean: {ratings_per_movie.mean():.1f}")
print(f"  Median: {ratings_per_movie.median():.1f}")
print(f"  Min: {ratings_per_movie.min()}")
print(f"  Max: {ratings_per_movie.max()}")


# ===================================
# 6. SAVE PROCESSED DATA
# ===================================
print("\n6. Saving processed data...")

ratings.to_csv('processed_ratings.csv', index=False)
users.to_csv('processed_users.csv', index=False)
movies.to_csv('processed_movies.csv', index=False)
rating_matrix.to_csv('rating_matrix.csv')

print("✓ Saved processed files")

print("\n" + "="*50)
print("DATA LOADING COMPLETE!")
print("="*50)
print("\nNext step: Run Phase 3 - ALS Matrix Factorization")
