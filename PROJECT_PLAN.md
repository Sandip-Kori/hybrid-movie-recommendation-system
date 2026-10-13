# Project Plan: Hybrid Movie Recommendation System

Based on the research paper:  
**"Design of an Efficient Hybrid Recommender System Using Item-Based Filtering and Machine Learning Models for Movie Recommendation"**  
*Authors: Rutvik Somnath Khake & Sandip Ghanshyam Kori (TIMSCDR, Mumbai)*

---

## 1. Project Overview & Scope
The objective is to implement a lightweight, fully working, and explainable hybrid movie recommendation application that pairs **Item-Based Collaborative Filtering (CF)** with **K-Nearest Neighbours (KNN)** rating prediction using an **alpha-weighted blending formula**:
$$\hat{r}_{\text{hybrid}} = \alpha \cdot \hat{r}_{\text{CF}} + (1 - \alpha) \cdot \hat{r}_{\text{KNN}}$$
where $\alpha \in [0, 1]$ is tuned on validation data (with $\alpha = 0.95$ serving as the empirical reference value from the paper).

The system includes:
- Clean data ingestion with MovieLens ratings and metadata.
- Strict train / validation / test splits to prevent data leakage.
- Clear mathematical fallbacks for sparse interaction regimes.
- Automated evaluation on held-out test data (RMSE, MAE, execution time).
- A modern Streamlit web UI styled with dark navy and purple/blue accents.
- Pytest automated test suite covering algorithms, edge cases, and formulas.

---

## 2. Project Architecture & File Organization
```
d:\MovieRcomendation\
├── .gitignore                    # Secrets, caches, artifacts exclusions
├── .env.example                  # Safe configuration placeholders
├── requirements.txt              # Core dependencies
├── PROJECT_PLAN.md               # Architecture and roadmap
├── TASK_TRACKER.md               # Ground-truth task tracking
├── CHANGELOG.md                  # Chronological change log
├── README.md                     # Comprehensive project documentation
├── app.py                        # Streamlit web application
├── src/
│   ├── __init__.py
│   ├── config.py                 # Constants, paths, default hyperparameters
│   ├── data_loader.py            # Dataset loading, preprocessing, split management
│   ├── models.py                 # ItemBasedCF, KNNRecommender, HybridRecommender
│   └── evaluation.py             # Validation alpha-tuning, test RMSE/MAE, comparison charts
└── tests/
    ├── __init__.py
    ├── test_data_loader.py       # Data split, shape, matrix construction tests
    ├── test_models.py            # CF, KNN, Hybrid formula, boundary, fallback tests
    └── test_evaluation.py        # Metric calculations and alpha tuning tests
```

---

## 3. Development Phases

### Phase 1: Foundation & Project Tracking Setup
- [x] Inspect research paper and design specifications.
- [x] Initialize Git repository.
- [x] Create `.gitignore`, `requirements.txt`, and `.env.example`.
- [x] Establish `PROJECT_PLAN.md`, `TASK_TRACKER.md`, `CHANGELOG.md`, `README.md`.

### Phase 2: Dataset Ingestion & Preprocessing (`src/data_loader.py`, `src/config.py`)
- Automated download/loading of MovieLens dataset (ratings & movies).
- Cleaning: removal of duplicates, invalid ratings.
- Chronological or reproducible seeded split: Train (80%), Validation (10%), Test (10%).
- Sparse user-item matrix creation with movie metadata indexing.
- Comprehensive data loader tests.

### Phase 3: Recommendation Algorithms (`src/models.py`)
- **Item-Based CF**: Item-item cosine similarity, k-nearest neighbours, rating prediction formula, documented fallbacks (item mean $\to$ user mean $\to$ global mean).
- **KNN Recommender**: Scikit-learn cosine nearest neighbours on item representation, rating prediction.
- **Hybrid Recommender**: Blends CF and KNN predictions via $\alpha \cdot \text{CF} + (1 - \alpha) \cdot \text{KNN}$; rating clipping to $[1.0, 5.0]$.
- **Top-N Item Recommendation**: Ranked recommendations given a target movie/user, excluding query item, with optional genre filtering.

### Phase 4: Model Evaluation & Alpha Tuning (`src/evaluation.py`)
- Alpha tuning grid search over $\alpha \in [0.10, 0.95]$ using validation split only (minimizing RMSE).
- Evaluation on held-out test split: compute RMSE, MAE, execution/fitting time for CF, KNN, and Hybrid.
- Simple Matplotlib visualization comparing RMSE, MAE, and fitting time.
- Verifiable test-leakage prevention.

### Phase 5: Automated Testing Suite (`tests/`)
- Unit and regression tests covering:
  - Preprocessing and split integrity.
  - CF mathematical prediction and fallback handling.
  - KNN prediction.
  - Hybrid formula weighting and boundary conditions ($\alpha = 0$, $\alpha = 1$, $\alpha = 0.95$).
  - Rating range clipping $[1.0, 5.0]$.
  - Evaluation metric calculation (RMSE, MAE).
  - Recommendation output sorting and exclusion of query item.

### Phase 6: Interactive Streamlit UI (`app.py`)
- Dark navy background (`#0f172a`), purple/blue accent colors (`#6366f1`, `#8b5cf6`).
- Navigation:
  - **Home**: Project context, dataset summary, architecture overview.
  - **Recommendations**: Movie search, genre filter, model selector (CF, KNN, Hybrid), alpha slider, recommendation count (3, 5, 10), card-based recommendation layout.
  - **Model Evaluation**: Test split metrics table (RMSE, MAE, training time), alpha tuning curve, comparison bar chart.
  - **About**: Research paper overview, mathematical formulation, student authors, limitations.

### Phase 7: Verification, Documentation & Security Review
- Full test suite execution with pytest.
- Streamlit application smoke test.
- Git status and secret security audit (ensuring no secrets, no large binaries staged).
- Final updates to `TASK_TRACKER.md`, `CHANGELOG.md`, and `README.md`.
