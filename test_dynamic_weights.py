import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from src.utils.config import HYBRID_CONFIG

def sigmoid(count, threshold, k):
    return 1.0 / (1.0 + np.exp(-k * (count - threshold)))

def test_weights():
    threshold = HYBRID_CONFIG['transition_threshold']
    k = HYBRID_CONFIG['steepness_k']
    
    print(f"Dynamic Weighting Config: Threshold={threshold}, k={k}")
    print("-" * 50)
    print(f"{'Ratings':<10} | {'ALS Weight':<12} | {'Demo Weight':<12} | {'Cont Weight':<12}")
    print("-" * 50)
    
    test_counts = [0, 1, 5, 10, 20, 50, 100, 200, 500]
    
    for count in test_counts:
        als_confidence = sigmoid(count, threshold, k)
        w_als = 0.1 + (0.7 * als_confidence)
        
        remaining = 1.0 - w_als
        w_demo = remaining * 0.5
        w_content = remaining * 0.5
        
        print(f"{count:<10} | {w_als:<12.4f} | {w_demo:<12.4f} | {w_content:<12.4f}")

if __name__ == "__main__":
    test_weights()
