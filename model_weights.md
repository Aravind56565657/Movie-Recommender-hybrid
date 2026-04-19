# Model Weights Configuration

The hybrid recommender system uses **static (fixed)** weights defined in the configuration file. These weights were determined based on the optimal values cited in the reference paper (initially derived via PSO - Particle Swarm Optimization).

## Configuration File
**Path:** `src/utils/config.py`

## 1. Hybrid Combination Weights
These weights determine the contribution of each individual model to the final hybrid prediction.

| Component | Weight Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| **ALS Model** | `weight_als` | **0.30** | Matrix Factorization (Collaborative Filtering) |
| **Demographic Model** | `weight_demographic` | **0.35** | User Similarity based on demographics |
| **Content Model** | `weight_content` | **0.35** | Item Similarity based on content features |

*Note: The sum of these weights is normalized to 1.0.*

## 2. Demographic Similarity Weights
Within the Demographic model, user similarity is calculated as a weighted sum of three factors.

| Factor | Weight Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| **Age** | `weight_age` | **0.31** | Similarity based on age groups |
| **Gender** | `weight_gender` | **0.27** | Similarity based on gender |
| **Genre** | `weight_genre` | **0.42** | Similarity based on movie genre preferences |

## Code Reference
Excerpt from `src/utils/config.py`:

```python
# ===================================
# DEMOGRAPHIC SIMILARITY (Step 2)
# ===================================
DEMOGRAPHIC_CONFIG = {
    # ...
    # Optimal weights from PSO (from paper)
    'weight_age': 0.31,          # L1
    'weight_gender': 0.27,       # L2
    'weight_genre': 0.42,        # L3
    # ...
}

# ===================================
# HYBRID WEIGHTING (Step 4)
# ===================================
HYBRID_CONFIG = {
    # Alternative: per-component weights
    'weight_als': 0.3,
    'weight_demographic': 0.35,
    'weight_content': 0.35,
}
```
