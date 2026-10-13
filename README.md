# Hybrid Movie Recommendation System

> **A Lightweight Research Implementation Combining Item-Based Collaborative Filtering and K-Nearest Neighbours**  
> Based on the research paper:  
> *"Design of an Efficient Hybrid Recommender System Using Item-Based Filtering and Machine Learning Models for Movie Recommendation"*  
> **Authors**: Rutvik Somnath Khake (Roll No: 69) & Sandip Ghanshyam Kori (Roll No: 78)  
> **Institution**: Department of MCA, Thakur Institute of Management Studies, Career Development & Research (TIMSCDR), Mumbai.

---

## 📌 1. Project Overview

Collaborative Filtering (CF) is the cornerstone of personalized recommendation engines. However, real-world movie platforms face critical challenges:
- **Matrix Sparsity**: Users rate only a tiny fraction of the expansive catalog.
- **Cold-Start & Thin Neighborhoods**: Sparse items lack enough co-ratings to form confident similarity neighborhoods.
- **Heavy Infrastructure Requirements**: Deep learning models often require dedicated GPU hardware unsuitable for lightweight deployment.

This mini-project demonstrates an **efficient, explainable hybrid recommender** that blends:
1. **Item-Based Collaborative Filtering (CF)** using cosine item-item similarities.
2. **K-Nearest Neighbours (KNN)** rating prediction to provide stable corrections in sparse regions.
3. **Automated $\alpha$-Weighted Blending**:
   $$\hat{r}_{\text{hybrid}} = \alpha \cdot \hat{r}_{\text{CF}} + (1 - \alpha) \cdot \hat{r}_{\text{KNN}}$$
   where $\alpha \in [0.10, 0.95]$ is tuned on an internal validation split to minimize Root Mean Squared Error (RMSE). In the reference research paper, $\alpha$ converged at **0.95**, demonstrating that CF provides the primary signal while KNN acts as a stabilizing fallback for thin neighborhoods.

---

## 🛠️ 2. Tech Stack

- **Language**: Python 3.11+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib
- **Web Interface**: Streamlit
- **Testing**: Pytest

---

## 📂 3. Repository Architecture

```
d:\MovieRcomendation\
├── .gitignore               # Strict exclusion of secrets, caches, and datasets
├── .env.example             # Safe placeholder template
├── requirements.txt         # Core dependencies
├── PROJECT_PLAN.md          # Multi-phase engineering roadmap
├── TASK_TRACKER.md          # Ground-truth verified task tracker
├── CHANGELOG.md             # Chronological project history
├── README.md                # Project documentation and user guide
├── app.py                   # Streamlit interactive application
├── src/
│   ├── config.py            # Central parameters, seeds, dataset URLs
│   ├── data_loader.py       # Dataset download, cleaning, train/val/test splits
│   ├── models.py            # ItemBasedCF, KNNRecommender, HybridRecommender
│   └── evaluation.py        # Alpha tuning, test RMSE/MAE, comparison charts
└── tests/
    ├── test_data_loader.py  # Tests for data splits and preprocessing
    ├── test_models.py       # Tests for CF, KNN, Hybrid, fallbacks, clipping
    └── test_evaluation.py   # Tests for evaluation metrics and alpha search
```

---

## 🚀 4. Installation & Quick Start

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd MovieRcomendation
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```bash
pytest -v
```

### Step 5: Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 5. Methodology & Mathematical Details

### A. Item-Based Collaborative Filtering (CF)
Computes cosine similarity between item rating vectors in the interaction matrix:
$$\text{sim}(i, j) = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\|_2 \|\mathbf{r}_j\|_2}$$
To predict rating $\hat{r}_{u, i}$ for user $u$ on movie $i$:
$$\hat{r}_{\text{CF}}(u, i) = \frac{\sum_{j \in N_k(i; u)} \text{sim}(i, j) \cdot r_{u, j}}{\sum_{j \in N_k(i; u)} |\text{sim}(i, j)|}$$
**Fallback hierarchy**: If user $u$ has not rated any of item $i$'s neighbors, the model falls back to the item mean rating $\bar{r}_i$, then user mean rating $\bar{r}_u$, and finally global mean rating $\mu$.

### B. K-Nearest Neighbours (KNN)
Uses Scikit-learn's `NearestNeighbors` fitted on item rating features with cosine distance to locate the $k$ most similar movies and estimate rating regression values.

### C. Hybrid Prediction Formula
$$\hat{r}_{\text{hybrid}}(u, i) = \alpha \cdot \hat{r}_{\text{CF}}(u, i) + (1 - \alpha) \cdot \hat{r}_{\text{KNN}}(u, i)$$
- All predicted ratings are strictly clipped to $[1.0, 5.0]$.
- $\alpha = 0.95$ is reported as the empirical reference value in the paper. The application allows users to dynamically slide $\alpha$ or use the validation-tuned optimum.

### D. Leak-Free Evaluation Split
- **Train Set (80%)**: Used to fit similarity matrices and neighbor structures.
- **Validation Set (10%)**: Used strictly to tune $\alpha \in [0.10, 0.95]$ via grid search minimizing RMSE.
- **Test Set (10%)**: Held-out split evaluated strictly once after model and parameter selection.

---

## ⚠️ 6. Assumptions & Limitations

1. **Dataset Scale**: Designed for educational and research demonstration on MovieLens datasets (e.g. 100k records). Industrial catalogs with millions of titles require distributed matrix factorization or Approximate Nearest Neighbor (ANN) indexes.
2. **Feature Scope**: The core hybrid focuses on rating interactions; metadata (genres, cast, tags) is used for filtering rather than content embedding.
3. **Alpha Grid Search**: Optimization uses a discrete grid rather than gradient descent.
