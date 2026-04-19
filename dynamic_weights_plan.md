# Plan: Dynamic Weighting for Hybrid Recommender

## 1. Objective
To mitigate the limitations of static weights by dynamically adjusting the contribution of each model component (ALS, Demographic, Content) based on **User Reliability** and **Data Availability**. This aims to improve personalization accuracy (RMSE, F1) and specifically address the "Cold Start" problem.

## 2. Core Concept: Confidence-Based Weighting
Static weights assume every user is the same. Dynamic weights recognize that:
- **ALS (Collaborative Filtering)** is powerful but requires a user to have many ratings (sparse data = poor predictions).
- **Demographic & Content Models** do not rely on a user's history but rather on their profile and metadata. These are more reliable for new users.

### The Strategy
We will implement a **User Confidence Score** based on the number of ratings a user has provided (`n_ratings`).
- **Low `n_ratings` (Cold Start)**: Shift weights towards **Demographic** and **Content**.
- **High `n_ratings` (Power User)**: Shift weights towards **ALS**.

## 3. Mathematical Model (Sigmoid Weighting)
Instead of hard thresholds, we use a smooth sigmoid function to transition weights dynamically.

Let $N_u$ be the number of ratings for user $u$.
Let $w_{als}(u)$ be the dynamic boost factor.

$$ w_{als}(u) = \frac{1}{1 + e^{-k(N_u - \text{threshold})}} $$

- **Threshold**: The "tipping point" (e.g., 50 ratings).
- **k**: Steepness of the transition.

### Adjusted Weights Calculation
For each user $u$:
1. Calculate Dynamic ALS Importance, normalized to a range $[min, max]$.
2. Calculate Residual Importance: $1.0 - w_{als}(u)$
3. Split Residual:
    - $w'_{demo} = w'_{residual} \times 0.5$
    - $w'_{content} = w'_{residual} \times 0.5$

## 4. Implementation Plan

### Step 1: Precompute User Statistics
- Modify `DataPreprocessor` to calculate and store `n_ratings` for every user.
- Persist this mapping (User ID -> Count) for lookup during inference.

### Step 2: Define Weighting Function
- Create a utility function `calculate_dynamic_weights(user_id, rating_count)` that implements the sigmoid logic.
- Allow configuration of `min_weight` and `max_weight` to prevent any model from vanishing completely.

### Step 3: Update Hybrid Recommender
- Modify `HybridRecommender.fit()` method.
- **Current State**: `final = w1*ALS + w2*Demo + w3*Content` (Scalar multiplication).
- **New State**:
    - Construct a weight matrix $W$ of shape `[n_users, 3]`.
    - Apply unique weights per user row during combination: `final = W[:, 0]*ALS + W[:, 1]*Demo + W[:, 2]*Content`.

### Step 4: Hyperparameter Tuning
- We need to tune the new hyperparameters:
    - `transition_threshold` (e.g., 20, 50, 100 ratings)
    - `steepness_k`
- Run the `evaluate_model` pipeline to see if RMSE improves compared to the static baseline.

## 5. Expected Improvements
| Metric | Expected Change | Reason |
| :--- | :--- | :--- |
| **RMSE (Overall)** | **Decrease (Better)** | Comparison error for low-data users will drop as we rely less on their noisy/sparse ALS predictions. |
| **Precision/Recall** | **Increase** | Recommendations for new users will be more relevant (based on demographics/content) which perform better than random/under-trained ALS. |
| **Personalization** | **High** | Users get recommendations adapted to their maturity level in the system. |