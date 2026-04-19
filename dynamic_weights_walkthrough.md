# Walkthrough: Dynamic Weighting Implementation

## 1. Goal
Upgrade the hybrid recommender system from strictly static weights to **dynamic weights** based on user rating history. This significantly improves personalization by trusting ALS more for power users and Demographic/Content models more for new users.

## 2. Key Architecture Changes

### A. Data Preprocessing
- **File:** `src/data/preprocessor.py`
- **Change:** 
    - Calculated `n_ratings` for each user during the preprocessing step.
    - Stored this metadata alongside the rating matrix to be accessible during model training.

### B. Weight Calculation Logic
- **File:** `src/utils/math_utils.py` (New File)
- **Change:** 
    - Implemented `calculate_dynamic_weights(n_ratings, config)`
    - Uses a **Sigmoid Function** to smoothly transition ALS weight from `0.1` (Cold Start) to `0.8` (Power User).
    - Formula: $w_{als} = \frac{1}{1 + e^{-k(N - N_0)}}$

### C. Hybrid Model Integration
- **File:** `src/models/hybrid_model.py`
- **Change:**
    - Modified `fit()` to accept user statistics.
    - Instead of scalar multiplication (`0.3 * ALS`), we now compute a **Weight Matrix** of shape `(n_users, 3)`.
    - Applied row-wise weighting: $P_{final}[u] = W[u,0] \cdot P_{als}[u] + W[u,1] \cdot P_{demo}[u] + W[u,2] \cdot P_{content}[u]$

## 3. Configuration Parameters
New parameters added to `src/utils/config.py`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `dynamic_weighting` | `True` | Toggle to enable/disable this feature |
| `min_als_weight` | `0.1` | Weight for users with 0 ratings |
| `max_als_weight` | `0.8` | Weight for users with many ratings |
| `transition_threshold` | `50` | Number of ratings where ALS becomes dominant |
| `steepness` | `0.1` | How fast the transition happens |

## 4. Expected Results & Verification
- **Cold Start Users (<20 ratings):** Will receive recommendations heavily influenced by their Age, Gender, and Genre preferences.
- **Power Users (>50 ratings):** Will receive highly specific Collaborative Filtering recommendations.
- **Overall Metric Impact:**
    - RMSE should decrease (improved accuracy).
    - Precision@10 should increase (better relevance).

## 5. Execution
To run the system with these new dynamic weights:
```bash
python main.py --mode full --dataset ml-100k --dynamic
```
