# Changelog

All notable changes to the Hybrid Movie Recommendation System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.1] - 2026-10-13

### Fixed
- **Sidebar Layout & Sizing**:
  - Corrected sidebar container flexbox from row orientation to vertical column (`flex-direction: column !important`), restoring full 320px width for all sidebar controls.
  - Resolved horizontal content squishing and awkward line breaks on navigation options (`🏠 Home`, `✨ Recommendations`, `📊 Model Evaluation`, `ℹ️ About`).
  - Added `white-space: nowrap` and clean alignment for Model Settings heading, branding, and navigation pills.
  - Ensured full width for recommendation model dropdown, alpha slider, CF/KNN badges, and count selectors.
  - Guaranteed vertical scrolling capability within the sidebar when viewport height is constrained.

## [1.0.0] - 2026-10-13

### Added
- **Core Architecture & Configuration**:
  - `src/config.py`: Established constants, hyperparameters ($k=20$, reference $\alpha=0.95$), data paths, rating boundaries ($[1.0, 5.0]$), and dark navy/purple UI theme tokens.
- **Dataset Ingestion & Validation**:
  - `src/data_loader.py`: Implemented MovieLens automatic downloading with graceful offline synthetic fallback generator; duplicate and invalid rating filtration; strict leak-free train (80%), validation (10%), and test (10%) splits; user-item interaction matrix and sparsity metrics.
- **Recommendation Algorithms**:
  - `src/models.py`:
    - `ItemBasedCF`: Cosine similarity computation on mean-centered interaction matrices, top-$k$ neighbor prediction, and tiered fallback logic (item mean $\to$ user mean $\to$ global mean).
    - `KNNRecommender`: Scikit-learn `NearestNeighbors` item regression with cosine distance to address sparse regions.
    - `HybridRecommender`: Alpha-weighted blending $\hat{r}_{\text{hybrid}} = \alpha \cdot \hat{r}_{\text{CF}} + (1 - \alpha) \cdot \hat{r}_{\text{KNN}}$ with strict $[1.0, 5.0]$ rating clipping and similarity ranking excluding the query movie.
- **Evaluation & Validation Alpha Search**:
  - `src/evaluation.py`: Implemented leak-free validation grid search for $\alpha \in [0.10, 0.95]$; RMSE and MAE calculation; multi-model comparative benchmarking on held-out test data; and Matplotlib comparative charts.
- **Interactive Streamlit Web Interface**:
  - `app.py`: Created an interactive UI styled in dark navy (`#0f172a`) with purple/blue gradients:
    - **Home**: Project context, dataset summary cards, catalog sparsity metric.
    - **Recommendations**: Searchable movie selectbox, genre filter, top-$N$ recommendation cards ($N \in \{3, 5, 10\}$) with similarity score badges and predicted rating stars.
    - **Model Evaluation**: Test-set comparison metrics table, validation tuning curve vs. paper reference $\alpha=0.95$, and side-by-side Matplotlib charts.
    - **About**: Research attribution to authors Rutvik Somnath Khake & Sandip Ghanshyam Kori (TIMSCDR, Mumbai), mathematical formulations, and project limitations.
- **Automated Test Suite**:
  - `tests/test_data_loader.py`: 5 tests covering data loading, cleaning, splits, and sparsity.
  - `tests/test_models.py`: 8 tests covering CF, KNN, hybrid blending, boundary $\alpha$, fallbacks, and clipping.
  - `tests/test_evaluation.py`: 4 tests covering RMSE/MAE math, evaluation metrics, and validation tuning.
  - `tests/test_app_integration.py`: 1 end-to-end integration smoke test.
  - Total: 18 passing tests with 100% pass rate.
- **Security & Git**:
  - `.gitignore` and `.env.example` verified; repository scanned with zero secrets detected.
