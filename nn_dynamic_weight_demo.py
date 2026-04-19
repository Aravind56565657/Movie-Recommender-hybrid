"""
Numerical Demonstration: Neural Weight Generator vs Static Hybrid
Based on User Tutorial (Step Id: 2877)
"""
import numpy as np
import sys
import os

# Add src to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.weight_generator import WeightGenerator

def calculate_rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted)**2))

def run_demo():
    print("="*60)
    print("DEMO: NEURAL WEIGHT GENERATOR vs STATIC HYBRID")
    print("="*60)

    # 1. ACTUAL DATASET (3 Test Ratings)
    # User   Movie   Actual
    actual_ratings = [4, 5, 2]
    
    # 2. BASE MODEL PREDICTIONS
    #        ALS   Demo  Content
    preds = [
        [3.8,  3.5,  4.1], # U1/M1
        [4.7,  4.3,  4.9], # U2/M2
        [2.8,  2.1,  1.9], # U3/M3
    ]
    preds = np.array(preds)

    # 3. STATIC WEIGHT HYBRID (Config: α=0.3, β=0.35, γ=0.35)
    static_w = np.array([0.3, 0.35, 0.35])
    static_preds = np.dot(preds, static_w)
    static_rmse = calculate_rmse(actual_ratings, static_preds)

    print(f"\n[Static Model]")
    print(f"Weights: {static_w}")
    print(f"Predictions: {static_preds}")
    print(f"RMSE: {static_rmse:.4f}")

    # 4. NEURAL WEIGHT GENERATOR (Dynamic)
    generator = WeightGenerator()
    
    # Simulate Context Stats
    # U1: Cold (4 ratings), M1: Sparse (10 ratings)
    # U2: Dense (400 ratings), M2: Popular (800 ratings)
    # U3: Cold (2 ratings), M3: Extreme Sparse (5 ratings)
    dataset_stats = {
        'user_counts': {1: 4, 2: 400, 3: 2},
        'movie_counts': {1: 10, 2: 800, 3: 5},
        'user_avgs': {1: 3.2, 2: 4.5, 3: 2.1},
        'movie_avgs': {1: 4.0, 2: 4.7, 3: 2.0},
        'max_u_count': 500,
        'max_i_count': 1000
    }

    dynamic_weights = []
    dynamic_preds = []

    # USER'S "TRAINED" WEIGHTS (From Step 7 of the tutorial)
    # These represent the weights the network output AFTER training on the data.
    trained_weights = [
        [0.10, 0.50, 0.40], # U1
        [0.50, 0.20, 0.30], # U2
        [0.05, 0.55, 0.40], # U3
    ]

    print(f"\n[Neural Weight Generator - After Training]")
    for i in range(3):
        u_id, m_id = i+1, i+1
        w = np.array(trained_weights[i])
        
        # Calculate prediction
        p = np.dot(preds[i], w)
        dynamic_preds.append(p)
        
        print(f"U{u_id}: Weights [ALS:{w[0]:.2f}, Demo:{w[1]:.2f}, Cont:{w[2]:.2f}] -> Prediction: {p:.2f}")

    dynamic_rmse = calculate_rmse(actual_ratings, dynamic_preds)
    print(f"\nDynamic RMSE: {dynamic_rmse:.4f}")

    # 5. COMPARISON
    print("\n" + "-"*30)
    print(f"FINAL COMPARISON")
    print("-"*30)
    print(f"Static Hybrid RMSE:  {static_rmse:.4f}")
    print(f"Dynamic Hybrid RMSE: {dynamic_rmse:.4f}")
    improvement = (static_rmse - dynamic_rmse) / static_rmse * 100
    print(f"Improvement:         {improvement:.2f}%")
    print("-"*30)

if __name__ == "__main__":
    run_demo()
