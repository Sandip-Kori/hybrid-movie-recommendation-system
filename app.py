"""Streamlit Web Application: Hybrid Movie Recommendation System."""

import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_K_NEIGHBORS,
    MAX_RATING,
    MIN_RATING,
    PAPER_REFERENCE_ALPHA,
)
from src.data_loader import (
    build_interaction_matrix,
    calculate_sparsity,
    clean_and_validate_data,
    download_movielens_dataset,
    get_movie_metadata_maps,
    split_ratings_data,
)
from src.evaluation import (
    compare_all_models,
    generate_comparison_chart,
    tune_alpha_on_validation,
)
from src.models import HybridRecommender, ItemBasedCF, KNNRecommender

# Page Configuration
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Navy & Purple/Blue Theme
st.markdown(
    """
    <style>
    /* Dark Navy Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    /* Headings & Purple Gradient Text */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
    .gradient-title {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Card Containers */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a855f7;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Recommendation Movie Card */
    .movie-card {
        background: linear-gradient(180deg, #1e293b 0%, #151d2f 100%);
        border: 1px solid #334155;
        border-left: 4px solid #8b5cf6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease-in-out, border-color 0.15s ease-in-out;
    }
    .movie-card:hover {
        border-color: #a855f7;
        transform: translateY(-2px);
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .movie-genres {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .movie-badge {
        display: inline-block;
        padding: 3px 8px;
        background-color: #312e81;
        color: #c7d2fe;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 6px;
    }
    .rating-badge {
        display: inline-block;
        padding: 3px 8px;
        background-color: #1e1b4b;
        color: #f59e0b;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
    }

    /* Primary Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """Load dataset, clean, split, and compute metadata maps."""
    raw_ratings, raw_movies = download_movielens_dataset()
    ratings_df, movies_df = clean_and_validate_data(raw_ratings, raw_movies)
    train_df, val_df, test_df = split_ratings_data(ratings_df, chronological=True)
    id_to_title, title_to_id, genres_list = get_movie_metadata_maps(movies_df)

    interaction_matrix = build_interaction_matrix(train_df)
    sparsity = calculate_sparsity(interaction_matrix)

    return {
        "ratings_df": ratings_df,
        "movies_df": movies_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "id_to_title": id_to_title,
        "title_to_id": title_to_id,
        "genres_list": genres_list,
        "sparsity": sparsity,
    }


@st.cache_resource(show_spinner=False)
def get_trained_models(
    _train_df: pd.DataFrame, _val_df: pd.DataFrame, _test_df: pd.DataFrame, k: int
):
    """Train models, tune alpha on validation, evaluate on test, and cache results."""
    results_df, best_alpha, val_tuning_history, hybrid_model = compare_all_models(
        _train_df, _val_df, _test_df, k_neighbors=k
    )

    # Standalone CF and KNN models for direct selection
    cf_model = hybrid_model.cf_model
    knn_model = hybrid_model.knn_model

    return {
        "results_df": results_df,
        "best_alpha": best_alpha,
        "val_tuning_history": val_tuning_history,
        "hybrid_model": hybrid_model,
        "cf_model": cf_model,
        "knn_model": knn_model,
    }


# Load Data
with st.spinner("Loading MovieLens dataset..."):
    data_bundle = load_and_prepare_data()

train_df = data_bundle["train_df"]
val_df = data_bundle["val_df"]
test_df = data_bundle["test_df"]
movies_df = data_bundle["movies_df"]
id_to_title = data_bundle["id_to_title"]
title_to_id = data_bundle["title_to_id"]
genres_list = data_bundle["genres_list"]

# Train Models
with st.spinner("Fitting recommender models & tuning alpha..."):
    models_bundle = get_trained_models(train_df, val_df, test_df, k=DEFAULT_K_NEIGHBORS)

hybrid_model = models_bundle["hybrid_model"]
cf_model = models_bundle["cf_model"]
knn_model = models_bundle["knn_model"]
best_alpha = models_bundle["best_alpha"]
results_df = models_bundle["results_df"]
val_tuning_history = models_bundle["val_tuning_history"]

# Sidebar Navigation & Settings
st.sidebar.markdown("## 🎬 Navigation")
app_mode = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "✨ Recommendations", "📊 Model Evaluation", "ℹ️ About"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Recommendation Model",
    ["Hybrid CF+KNN", "Item-Based CF", "Standalone KNN"],
    index=0,
)

# Alpha Control for Hybrid Model
current_alpha = PAPER_REFERENCE_ALPHA
if selected_model_name == "Hybrid CF+KNN":
    col_a, col_b = st.sidebar.columns([3, 1])
    use_tuned = col_b.button("Use Best α", help=f"Set to validation-tuned α = {best_alpha}")

    default_slider_alpha = best_alpha if use_tuned else PAPER_REFERENCE_ALPHA
    current_alpha = st.sidebar.slider(
        "Alpha Blending Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=float(default_slider_alpha),
        step=0.05,
        help="α = CF weight, (1 - α) = KNN weight. Reference: 0.95",
    )
    hybrid_model.set_alpha(current_alpha)

    st.sidebar.caption(
        f"**Weights**: CF = `{current_alpha*100:.0f}%` | KNN = `{(1-current_alpha)*100:.0f}%`"
    )

top_n_choice = st.sidebar.selectbox("Recommendation Count", [3, 5, 10], index=1)
genre_filter_choice = st.sidebar.selectbox("Genre Filter", ["All"] + genres_list, index=0)

# -------------------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------------------
if app_mode == "🏠 Home":
    st.markdown('<div class="gradient-title">Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A Research Implementation Combining Item-Based Collaborative Filtering and K-Nearest Neighbours</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Welcome to the **Hybrid Movie Recommender**, built according to the research methodology of:  
        *Design of an Efficient Hybrid Recommender System Using Item-Based Filtering and Machine Learning Models for Movie Recommendation*  
        by **Rutvik Somnath Khake** and **Sandip Ghanshyam Kori** (*TIMSCDR, Mumbai*).
        
        This system solves matrix sparsity and cold-start boundaries in movie catalogs by intelligently blending 
        an **Item-Based Collaborative Filtering** similarity network with a **K-Nearest Neighbours (KNN)** machine learning model.
        """
    )

    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Movies</div>
                <div class="metric-val">{len(movies_df):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Ratings</div>
                <div class="metric-val">{len(data_bundle['ratings_df']):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Matrix Sparsity</div>
                <div class="metric-val">{data_bundle['sparsity']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Train / Val / Test</div>
                <div class="metric-val">80 / 10 / 10%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🚀 Quick Navigation")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.info("🎯 **Recommendations Tab**: Pick any movie from the catalog, adjust model & alpha weights, and generate top-N suggestions.")
    with col_nav2:
        st.success("📈 **Model Evaluation Tab**: Inspect test-set RMSE/MAE benchmarks, execution times, and alpha validation tuning curves.")

# -------------------------------------------------------------
# PAGE 2: RECOMMENDATIONS
# -------------------------------------------------------------
elif app_mode == "✨ Recommendations":
    st.markdown('<div class="gradient-title">Discover Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-title">Active Model: <b>{selected_model_name}</b> '
        + (f'(α = {current_alpha:.2f})' if selected_model_name == 'Hybrid CF+KNN' else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    # Movie Selector
    movie_titles = sorted(list(title_to_id.keys()))
    default_movie = "Toy Story (1995)" if "Toy Story (1995)" in movie_titles else movie_titles[0]
    default_idx = movie_titles.index(default_movie)

    selected_title = st.selectbox(
        "Select or Search a Movie:",
        options=movie_titles,
        index=default_idx,
        help="Type or select a title from the catalog.",
    )

    selected_movie_id = title_to_id[selected_title]
    selected_row = movies_df[movies_df["movieId"] == selected_movie_id].iloc[0]

    st.markdown(
        f"**Selected Title**: `{selected_title}` | **Genres**: `{selected_row.get('genres', 'N/A')}` | **Movie ID**: `{selected_movie_id}`"
    )

    btn_recommend = st.button("Generate Recommendations", type="primary")

    if btn_recommend or "last_rec_movie" not in st.session_state:
        st.session_state["last_rec_movie"] = selected_movie_id

        # Determine which model to run
        if selected_model_name == "Item-Based CF":
            active_model = cf_model
        elif selected_model_name == "Standalone KNN":
            active_model = knn_model
        else:
            active_model = hybrid_model

        with st.spinner("Generating recommendations..."):
            recs = active_model.recommend_for_movie(
                movie_id=selected_movie_id,
                top_n=top_n_choice,
                genre_filter=genre_filter_choice,
                movies_df=movies_df,
            )

        st.markdown(f"### Top-{len(recs)} Recommended Movies")

        if not recs:
            st.warning("No recommendations found matching the selected genre criteria.")
        else:
            for rank, item in enumerate(recs, 1):
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">#{rank} {item['title']}</div>
                        <div class="movie-genres">🎭 {item['genres']}</div>
                        <div>
                            <span class="movie-badge">Similarity: {item['similarity']:.4f}</span>
                            <span class="rating-badge">★ Est. Rating: {item['predicted_rating']:.2f} / 5.0</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -------------------------------------------------------------
# PAGE 3: MODEL EVALUATION
# -------------------------------------------------------------
elif app_mode == "📊 Model Evaluation":
    st.markdown('<div class="gradient-title">Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evaluating on Held-Out Test Data (Zero Data Leakage)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Following the research methodology, all models are evaluated on the exact same **held-out test split (10%)**.
        Hyperparameter $\\alpha$ was tuned strictly on an internal **validation split (10%)** without touching the test records.
        """
    )

    # Metrics Summary Cards
    st.markdown("### 🏆 Test Set Comparison Summary")
    st.dataframe(
        results_df.style.format(
            {
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "Train Time (s)": "{:.4f}",
                "Eval Time (s)": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    # Matplotlib Comparison Figure
    st.markdown("### 📊 Benchmark Visualizations")
    fig = generate_comparison_chart(results_df)
    st.pyplot(fig)
    plt.close(fig)

    # Validation Alpha Tuning Curve
    st.markdown("### 🎛️ Alpha Blending Optimization on Validation Data")
    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        fig_alpha, ax_alpha = plt.subplots(figsize=(8, 3.5))
        fig_alpha.patch.set_facecolor("#0f172a")
        ax_alpha.set_facecolor("#1e293b")

        ax_alpha.plot(
            val_tuning_history["alpha"],
            val_tuning_history["val_rmse"],
            marker="o",
            color="#a855f7",
            linewidth=2,
            label="Validation RMSE",
        )
        ax_alpha.axvline(
            x=best_alpha,
            color="#38bdf8",
            linestyle="--",
            label=f"Optimal α = {best_alpha}",
        )
        ax_alpha.axvline(
            x=PAPER_REFERENCE_ALPHA,
            color="#f59e0b",
            linestyle=":",
            label=f"Paper Reference α = {PAPER_REFERENCE_ALPHA}",
        )

        ax_alpha.set_title("Validation RMSE vs. Alpha (α)", color="#f8fafc", fontweight="bold")
        ax_alpha.set_xlabel("Alpha (Weight for Item-Based CF)", color="#cbd5e1")
        ax_alpha.set_ylabel("Validation RMSE", color="#cbd5e1")
        ax_alpha.tick_params(colors="#cbd5e1")
        ax_alpha.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax_alpha.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        plt.tight_layout()

        st.pyplot(fig_alpha)
        plt.close(fig_alpha)

    with col_t2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Tuned Alpha (Val)</div>
                <div class="metric-val">{best_alpha}</div>
            </div>
            <div class="metric-card">
                <div class="metric-lbl">Paper Ref Alpha</div>
                <div class="metric-val">{PAPER_REFERENCE_ALPHA}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "An alpha close to 0.95 indicates that the Item-Based CF signal provides the primary predictive accuracy, "
            "while the KNN model supplies crucial stability and fills gaps for sparse neighborhoods."
        )

# -------------------------------------------------------------
# PAGE 4: ABOUT
# -------------------------------------------------------------
elif app_mode == "ℹ️ About":
    st.markdown('<div class="gradient-title">About the Research Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theoretical Foundations, Mathematical Modeling & Architecture</div>', unsafe_allow_html=True)

    st.markdown("### 📄 Research Citation")
    st.info(
        """
        **Paper Title**: *Design of an Efficient Hybrid Recommender System Using Item-Based Filtering and Machine Learning Models for Movie Recommendation*  
        **Authors**: Rutvik Somnath Khake (Roll No: 69) & Sandip Ghanshyam Kori (Roll No: 78)  
        **Academic Department**: Department of Master of Computer Applications (MCA)  
        **Institution**: Thakur Institute of Management Studies, Career Development & Research (TIMSCDR), Mumbai.
        """
    )

    st.markdown("### 📐 Mathematical Formulation")
    st.markdown(
        r"""
        #### 1. Item-Based Collaborative Filtering (CF)
        The similarity between two movies $i$ and $j$ is calculated via cosine similarity over mean-centered user rating vectors:
        $$\text{sim}(i, j) = \frac{\mathbf{u}_i \cdot \mathbf{u}_j}{\|\mathbf{u}_i\|_2 \|\mathbf{u}_j\|_2}$$
        
        The predicted rating $\hat{r}_{\text{CF}}(u, i)$ for user $u$ on movie $i$ is:
        $$\hat{r}_{\text{CF}}(u, i) = \frac{\sum_{j \in N_k(i; u)} \text{sim}(i, j) \cdot r_{u, j}}{\sum_{j \in N_k(i; u)} |\text{sim}(i, j)|}$$
        where $N_k(i; u)$ represents the $k$ most similar items to $i$ that user $u$ has rated.

        #### 2. K-Nearest Neighbours (KNN)
        Nearest neighbors are discovered in item rating feature space using Scikit-Learn's `NearestNeighbors` under cosine metric. KNN estimates the rating using neighbor distance weighting, preventing reliance on global averages.

        #### 3. Alpha-Weighted Hybrid Blending
        $$\hat{r}_{\text{hybrid}}(u, i) = \alpha \cdot \hat{r}_{\text{CF}}(u, i) + (1 - \alpha) \cdot \hat{r}_{\text{KNN}}(u, i)$$
        Predictions are clipped to the valid rating range:
        $$\hat{r}_{\text{final}} = \min(\max(\hat{r}_{\text{hybrid}}, 1.0), 5.0)$$
        """
    )

    st.markdown("### ⚠️ Assumptions & Limitations")
    st.markdown(
        """
        1. **Catalog Scale**: Evaluated on MovieLens benchmark scale. Commercial streaming services require distributed matrix decomposition or HNSW indexers.
        2. **Cold-Start Users**: For completely new users with 0 interactions, the system gracefully falls back to item and global averages.
        3. **Optimization Scope**: Alpha is tuned via grid search over candidate intervals rather than gradient optimization.
        """
    )
