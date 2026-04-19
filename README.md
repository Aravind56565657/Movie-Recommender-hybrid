
# Hybrid Movie Recommender System

A movie recommender system that combines Collaborative Filtering (ALS), Demographic Filtering, and Content-based Filtering to provide personalized movie recommendations.

## Prerequisites

- Python 3.8 or higher
- pip

## Installation

1.  Clone the repository (if applicable) or navigate to the project directory.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Project Setup

Before running the recommender system, you need to set up the project structure and load the data.

1.  **Data Source**: The setup script expects the MovieLens 100k dataset to be located at `C:\Users\aravi\OneDrive\Desktop\ml-100k`. Ensure this directory exists and contains the dataset files (`u.data`, `u.item`, etc.). If your data is in a different location, open `setup_structure.py` and update the `source_data` variable.

2.  **Run Setup Script**:
    Run the following command to create the necessary directories and copy the data:

    ```bash
    python setup_structure.py
    ```

    This will create directories like `data/`, `src/`, `models/`, `results/`, etc., and copy the data to `data/raw/ml-100k`.

3.  **Verify Data Loading**:
    You can run the test setup script to verify that the data is loaded correctly and to see some basic statistics:

    ```bash
    python test_setup.py
    ```

## Usage

The main entry point for the application is `main.py`. You can run the pipeline in different modes.

### Run the Full Pipeline

To run the complete recommendation pipeline (Data Preprocessing -> ALS -> Demographic -> Content -> Hybrid -> Evaluation):

```bash
python main.py --mode full
```

### Run Specific Modes

You can also run specific parts of the pipeline using the `--mode` argument:

-   `data`: Load and preprocess data only.
-   `als`: Train and predict using ALS (Collaborative Filtering).
-   `demographic`: Train and predict using Demographic Filtering.
-   `content`: Train and predict using Content-based Filtering.
-   `full`: Run everything.

Example:

```bash
python main.py --mode data
python main.py --mode als
```

## Project Structure

-   `main.py`: Main entry point.
-   `setup_structure.py`: Script to set up directories and data.
-   `test_setup.py`: Script to test data loading and exploration.
-   `src/`: Source code for data processing, models, and evaluation.
-   `data/`: Directory for raw and processed data.
-   `experiments/`: Directory for experiment logs.
-   `results/`: Directory for model results and plots.
-   `notebooks/`: Jupyter notebooks for analysis.
