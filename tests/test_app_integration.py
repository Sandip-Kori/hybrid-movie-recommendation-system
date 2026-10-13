"""Integration and smoke tests for the Streamlit application pipeline."""

import pandas as pd
import pytest

from src.data_loader import (
    build_interaction_matrix,
    calculate_sparsity,
    clean_and_validate_data,
    create_sample_dataset,
    get_movie_metadata_maps,
    split_ratings_data,
)
from src.evaluation import compare_all_models, generate_comparison_chart
from src.models import HybridRecommender, ItemBasedCF, KNNRecommender


def test_app_pipeline_smoke():
    # 1. Ingest data
    ratings_df, movies_df = create_sample_dataset()
    clean_ratings, clean_movies = clean_and_validate_data(ratings_df, movies_df)
    train_df, val_df, test_df = split_ratings_data(clean_ratings, chronological=True)

    # 2. Extract maps & metadata
    id_to_title, title_to_id, genres = get_movie_metadata_maps(clean_movies)
    assert len(id_to_title) > 0
    assert len(genres) > 0

    # 3. Sparsity check
    matrix = build_interaction_matrix(train_df)
    sparsity = calculate_sparsity(matrix)
    assert 0.0 <= sparsity <= 1.0

    # 4. Compare all models & tune alpha
    results_df, best_alpha, val_history, hybrid_model = compare_all_models(
        train_df, val_df, test_df, k_neighbors=5
    )
    assert not results_df.empty
    assert 0.0 <= best_alpha <= 1.0

    # 5. Recommendation generation for movie
    sample_movie_id = clean_movies.iloc[0]["movieId"]
    recs = hybrid_model.recommend_for_movie(
        movie_id=sample_movie_id, top_n=5, movies_df=clean_movies
    )
    assert len(recs) <= 5
    for r in recs:
        assert r["movieId"] != sample_movie_id

    # 6. Chart generation
    fig = generate_comparison_chart(results_df)
    assert fig is not None
