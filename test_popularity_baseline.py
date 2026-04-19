import numpy as np
import pandas as pd
from src.data.loader import load_movielens_1m_data
from src.utils.config import RAW_DATA_DIR_1M
from sklearn.model_selection import train_test_split

def test_pop():
    ratings, users, movies = load_movielens_1m_data(RAW_DATA_DIR_1M)
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42, stratify=ratings['user_id']
    )
    
    # Calculate popularity-based items (Global Mean + Item Bias)
    global_mean = train_ratings['rating'].mean()
    item_stats = train_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    
    # Basic Item Bias: mean - global_mean
    # Add smoothing to biases
    lambda_reg = 10
    item_biases = {}
    for mid, row in item_stats.iterrows():
        # Bayesian mean: (Sum + lambda * global) / (count + lambda)
        biased_mean = (row['mean'] * row['count'] + lambda_reg * global_mean) / (row['count'] + lambda_reg)
        item_biases[mid] = biased_mean
    
    # Calculate user rating counts from training
    user_counts = train_ratings.groupby('user_id').size().to_dict()
    
    results = []
    for _, row in test_ratings.iterrows():
        u = int(row['user_id'])
        m = int(row['movie_id'])
        actual = row['rating']
        
        count = user_counts.get(u, 0)
        group = "Cold" if count < 20 else ("Expert" if count > 100 else "Regular")
        
        # Popularity prediction
        pred = item_biases.get(m, global_mean)
        
        results.append({
            'group': group,
            'err': (actual - pred)**2
        })
    
    df = pd.DataFrame(results)
    summary = df.groupby('group').agg({
        'err': lambda x: np.sqrt(x.mean())
    })
    
    print("\nPure Popularity RMSE (Smoothed):")
    print(summary)

if __name__ == "__main__":
    test_pop()
