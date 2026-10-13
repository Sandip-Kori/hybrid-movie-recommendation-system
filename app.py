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

# Custom CSS for Dark Navy & Purple/Blue Theme Matching Reference Screenshot
st.markdown(
    """
    <style>
    /* 1. Global Dark Navy Canvas */
    .stApp {
        background-color: #060b17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* 2. Streamlit Header Chrome Styling (Eliminate white strip & menu without breaking sidebar) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        z-index: 50 !important;
    }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stAppDeployButton"] { display: none !important; }
    button[data-testid="stDeployButton"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }

    /* 3. Main Container Spacing & Layout */
    .main .block-container,
    div[data-testid="stMainBlockContainer"] {
        padding-top: 1.6rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1220px !important;
    }

    /* 4. Permanently Visible Fixed Left Sidebar */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="false"],
    section[data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        transform: none !important;
        left: 0px !important;
        margin-left: 0px !important;
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
        flex: 0 0 320px !important;
        flex-shrink: 0 !important;
        background-color: #0b1120 !important;
        border-right: 1px solid #16213b !important;
        padding: 0 !important;
        z-index: 99 !important;
        transition: none !important;
    }

    /* Scrollable vertical column for sidebar contents */
    div[data-testid="stSidebarContent"] {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 1.2rem 1.25rem 2rem 1.25rem !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stSidebarUserContent"] {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        flex: 1 1 auto !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    /* Ensure all widgets and vertical blocks in sidebar span full 100% width */
    div[data-testid="stSidebarUserContent"] > div,
    div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Hide collapse and expand toggle chevrons to keep sidebar permanently docked */
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] {
        display: none !important;
    }

    /* 5. Navigation Radio Items & Active Solid Purple Pill */
    div[data-testid="stRadio"] > label {
        color: #64748b !important;
        font-weight: 700;
        font-size: 0.76rem;
        text-transform: capitalize;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        margin-bottom: 4px !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer !important;
    }
    /* Hide the radio button circle bullet for clean pill look */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #4f46e5 !important;
        border: 1px solid #6366f1 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #16213b !important;
        color: #ffffff !important;
    }

    /* 6. Selectbox & Dropdown Dark Theme */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    /* 7. Slider Styling in Unified Purple Palette */
    div[data-testid="stSlider"] label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.88rem;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #8b5cf6 !important;
        border-color: #c084fc !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5) !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background-color: #6366f1 !important;
    }

    /* 8. Hero Card on Home Dashboard */
    .hero-card {
        background: linear-gradient(135deg, #0d1527 0%, #080f1d 100%);
        border: 1px solid #1a2744;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .hero-top {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 18px;
    }
    .hero-icon-box {
        font-size: 1.8rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
        flex-shrink: 0;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 45%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 2px;
    }
    .hero-objective {
        border-top: 1px solid #16213b;
        padding-top: 16px;
    }
    .objective-heading {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .objective-text {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.55;
    }

    /* 9. Dataset Summary Metric Cards */
    /* Equal Height Columns Container */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="column"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* 9. Dataset Summary Metric Cards */
    .metric-card-box {
        background: linear-gradient(180deg, #0d1527 0%, #09101f 100%);
        border: 1px solid #1a2744;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        min-height: 175px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card-box:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .metric-circle-icon {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        margin-bottom: 12px;
    }
    .metric-lbl-text {
        font-size: 0.76rem;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .metric-val-movies {
        font-size: 1.75rem;
        font-weight: 800;
        color: #c084fc;
        line-height: 1.25;
    }
    .metric-val-ratings {
        font-size: 1.75rem;
        font-weight: 800;
        color: #60a5fa;
        line-height: 1.25;
    }
    .metric-val-sparsity {
        font-size: 1.75rem;
        font-weight: 800;
        color: #2dd4bf;
        line-height: 1.25;
    }
    .metric-val-split {
        font-size: 1.3rem;
        font-weight: 800;
        color: #818cf8;
        line-height: 1.25;
        white-space: nowrap;
    }
    .metric-sub-text {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* 10. Quick Navigation Cards */
    .quick-nav-container {
        background: linear-gradient(180deg, #0d1527 0%, #09101f 100%);
        border: 1px solid #1a2744;
        border-radius: 12px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .quick-nav-header {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 18px;
    }
    .quick-nav-icon-circle {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .quick-nav-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .quick-nav-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* 11. Recommendation Movie Card */
    .movie-card {
        background: linear-gradient(180deg, #0d1527 0%, #09101f 100%);
        border: 1px solid #1a2744;
        border-left: 4px solid #8b5cf6;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
        transition: transform 0.15s ease-in-out, border-color 0.15s ease-in-out;
    }
    .movie-card:hover {
        border-color: #a855f7;
        transform: translateY(-2px);
    }
    .movie-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .movie-genres {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 10px;
    }
    .movie-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #1e1b4b;
        color: #c7d2fe;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 8px;
        border: 1px solid #312e81;
    }
    .rating-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #1e1b4b;
        color: #fbbf24;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 4px;
        border: 1px solid #312e81;
    }

    /* 12. Citation Box */
    .citation-box {
        background: linear-gradient(135deg, #0d1527 0%, #09101f 100%);
        border: 1px solid #1a2744;
        border-left: 4px solid #8b5cf6;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .citation-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .citation-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    /* 13. Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
    }
    div.stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: #ffffff !important;
    }

    /* 14. Section Headings & Metric Mini-Cards */
    .gradient-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 45%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.25;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.98rem;
        font-weight: 500;
        margin-bottom: 22px;
    }
    .metric-card {
        background: linear-gradient(180deg, #0d1527 0%, #09101f 100%);
        border: 1px solid #1a2744;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-val {
        color: #38bdf8;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.2;
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

# Navigation state synchronization
nav_options = ["🏠 Home", "✨ Recommendations", "📊 Model Evaluation", "ℹ️ About"]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🏠 Home"

if st.session_state["current_page"] not in nav_options:
    st.session_state["current_page"] = "🏠 Home"

# Sidebar Brand
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 22px; padding: 4px 0 16px 0; border-bottom: 1px solid #16213b;">
        <div style="font-size: 1.5rem; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border-radius: 10px; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35); flex-shrink: 0;">🎬</div>
        <div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; line-height: 1.25; white-space: nowrap;">Hybrid Movie</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; line-height: 1.25; white-space: nowrap;">Recommendation System</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div style="font-size: 0.8rem; font-weight: 600; color: #64748b; margin-bottom: 6px;">Navigation</div>',
    unsafe_allow_html=True,
)

app_mode = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=nav_options.index(st.session_state["current_page"]),
    label_visibility="collapsed",
)
st.session_state["current_page"] = app_mode

st.sidebar.markdown(
    '<div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 24px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; white-space: nowrap;">⚙️ Model Settings</div>',
    unsafe_allow_html=True,
)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    ["Hybrid CF + KNN", "Item-Based CF", "Standalone KNN"],
    index=0,
)

# Alpha Control for Hybrid Model
current_alpha = PAPER_REFERENCE_ALPHA
if selected_model_name in ["Hybrid CF + KNN", "Hybrid CF+KNN"]:
    current_alpha = st.sidebar.slider(
        "Alpha (Hybrid Weight)",
        min_value=0.0,
        max_value=1.0,
        value=float(PAPER_REFERENCE_ALPHA),
        step=0.05,
        help="Weight parameter α balancing Collaborative Filtering and KNN (Reference value: 0.95)",
    )
    hybrid_model.set_alpha(current_alpha)

    cf_pct = int(round(current_alpha * 100))
    knn_pct = int(round((1.0 - current_alpha) * 100))

    st.sidebar.markdown(
        f"""
        <div style="display: flex; gap: 10px; margin-top: 4px; margin-bottom: 16px;">
            <div style="flex: 1; text-align: center; background: #1e1b4b; border: 1px solid #4338ca; color: #c7d2fe; padding: 7px 10px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;">
                CF {cf_pct}%
            </div>
            <div style="flex: 1; text-align: center; background: #0c2d48; border: 1px solid #0284c7; color: #38bdf8; padding: 7px 10px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;">
                KNN {knn_pct}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

top_n_choice = st.sidebar.selectbox("Recommendation Count", [3, 5, 10], index=1)
genre_filter_choice = st.sidebar.selectbox("Genre Filter", ["All"] + genres_list, index=0)

st.sidebar.markdown(
    """
    <div style="margin-top: 36px; padding-top: 18px; border-top: 1px solid #16213b; display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.4rem;">🎓</span>
        <div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #94a3b8;">Research Mini Project</div>
            <div style="font-size: 0.72rem; color: #64748b;">Thakur Institute of Management Studies</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------------------
if app_mode == "🏠 Home":
    # Hero Card
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-top">
                <div class="hero-icon-box">🎬</div>
                <div>
                    <div class="hero-title">Hybrid Movie Recommendation System</div>
                    <div class="hero-subtitle">Item-Based Collaborative Filtering + KNN</div>
                </div>
            </div>
            <div class="hero-objective">
                <div class="objective-heading">🎯 Research Objective</div>
                <div class="objective-text">
                    To design an efficient hybrid recommender system using Item-Based Filtering and Machine Learning Models for movie recommendation by combining the strengths of Collaborative Filtering and K-Nearest Neighbours (KNN) using an alpha-weighted approach.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-circle-icon" style="background: rgba(168, 85, 247, 0.2); color: #c084fc;">🎞️</div>
                <div class="metric-lbl-text">TOTAL MOVIES</div>
                <div class="metric-val-movies">{len(movies_df):,}</div>
                <div class="metric-sub-text">(loaded MovieLens data)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-circle-icon" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">👥</div>
                <div class="metric-lbl-text">TOTAL RATINGS</div>
                <div class="metric-val-ratings">{len(data_bundle['ratings_df']):,}</div>
                <div class="metric-sub-text">(loaded MovieLens data)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-circle-icon" style="background: rgba(45, 212, 191, 0.2); color: #2dd4bf;">🕸️</div>
                <div class="metric-lbl-text">MATRIX SPARSITY</div>
                <div class="metric-val-sparsity">{data_bundle['sparsity']*100:.1f}%</div>
                <div class="metric-sub-text">(loaded MovieLens data)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-circle-icon" style="background: rgba(99, 102, 241, 0.2); color: #818cf8;">💾</div>
                <div class="metric-lbl-text">TRAIN / VALIDATION / TEST</div>
                <div class="metric-val-split">80% / 10% / 10%</div>
                <div class="metric-sub-text">(reproducible split)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick Navigation Section
    st.markdown(
        """
        <div style="margin-top: 30px; margin-bottom: 16px;">
            <div style="font-size: 1.45rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                <span>🚀</span> Quick Navigation
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 2px;">
                Explore the features of the Hybrid Movie Recommendation System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            """
            <div class="quick-nav-container">
                <div class="quick-nav-header">
                    <div class="quick-nav-icon-circle" style="background: rgba(168, 85, 247, 0.2); color: #c084fc;">🔍</div>
                    <div>
                        <div class="quick-nav-title">Explore Recommendations</div>
                        <div class="quick-nav-desc">Search for a movie and get personalized recommendations using our hybrid model.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Recommendations ➔", key="btn_nav_rec", type="primary"):
            st.session_state["current_page"] = "✨ Recommendations"
            st.rerun()

    with col_q2:
        st.markdown(
            """
            <div class="quick-nav-container">
                <div class="quick-nav-header">
                    <div class="quick-nav-icon-circle" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">📊</div>
                    <div>
                        <div class="quick-nav-title">Evaluate Models</div>
                        <div class="quick-nav-desc">Compare CF, KNN and Hybrid models using RMSE, MAE and training time metrics.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Model Evaluation ➔", key="btn_nav_eval", type="secondary"):
            st.session_state["current_page"] = "📊 Model Evaluation"
            st.rerun()

# -------------------------------------------------------------
# PAGE 2: RECOMMENDATIONS
# -------------------------------------------------------------
elif app_mode == "✨ Recommendations":
    st.markdown('<div class="gradient-title">Discover Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-title">Active Model: <b>{selected_model_name}</b> '
        + (f'(α = {current_alpha:.2f})' if "Hybrid" in selected_model_name else '')
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
    st.markdown(
        """
        <div class="citation-box">
            <div class="citation-title">Research Paper Publication</div>
            <div class="citation-desc">
                <b>Title</b>: <i>Design of an Efficient Hybrid Recommender System Using Item-Based Filtering and Machine Learning Models for Movie Recommendation</i><br>
                <b>Authors</b>: Rutvik Somnath Khake (Roll No: 69) &amp; Sandip Ghanshyam Kori (Roll No: 78)<br>
                <b>Department</b>: Master of Computer Applications (MCA)<br>
                <b>Institution</b>: Thakur Institute of Management Studies, Career Development &amp; Research (TIMSCDR), Mumbai.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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