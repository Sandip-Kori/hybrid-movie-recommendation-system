"""Unit tests for dataset loading, preprocessing, and splitting."""

import pandas as pd
import pytest

from src.config import MAX_RATING, MIN_RATING
from src.data_loader import (
    build_interaction_matrix,
    calculate_sparsity,
    clean_and_validate_data,
    create_sample_dataset,
    get_movie_metadata_maps,
    split_ratings_data,
)


def test_create_sample_dataset():
    ratings_df, movies_df = create_sample_dataset()
    assert isinstance(ratings_df, pd.DataFrame)
    assert isinstance(movies_df, pd.DataFrame)
    assert len(ratings_df) > 0
    assert len(movies_df) > 0
    assert set(["userId", "movieId", "rating", "timestamp"]).issubset(ratings_df.columns)
    assert set(["movieId", "title", "genres"]).issubset(movies_df.columns)


def test_clean_and_validate_data():
    raw_ratings = pd.DataFrame(
        [
            (1, 10, 4.0, 100),
            (1, 10, 4.5, 101),  # duplicate
            (2, 20, 6.0, 102),  # invalid rating > 5.0
            (3, 30, 0.5, 103),  # invalid rating < 1.0
            (4, 999, 3.5, 104),  # unknown movie
            (5, 10, None, 105),  # null rating
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )

    raw_movies = pd.DataFrame(
        [
            (10, "Movie A", "Action|Sci-Fi"),
            (20, "Movie B", "Comedy"),
            (30, "Movie C", "Drama"),
            (10, "Movie A Duplicate", "Action"),  # duplicate movieId
        ],
        columns=["movieId", "title", "genres"],
    )

    clean_ratings, clean_movies = clean_and_validate_data(raw_ratings, raw_movies)

    assert len(clean_movies) == 3
    assert len(clean_ratings) == 1
    assert clean_ratings.iloc[0]["userId"] == 1
    assert clean_ratings.iloc[0]["movieId"] == 10
    assert clean_ratings.iloc[0]["rating"] == 4.0
    assert clean_ratings["rating"].min() >= MIN_RATING
    assert clean_ratings["rating"].max() <= MAX_RATING


def test_split_ratings_data():
    ratings_df, _ = create_sample_dataset()
    train_df, val_df, test_df = split_ratings_data(
        ratings_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, chronological=True
    )

    total_len = len(ratings_df)
    assert len(train_df) + len(val_df) + len(test_df) == total_len
    assert len(train_df) == int(total_len * 0.8)
    assert len(val_df) == int(total_len * 0.1)

    # Check chronological ordering: max train timestamp <= min test timestamp
    assert train_df["timestamp"].max() <= test_df["timestamp"].min()


def test_build_interaction_matrix_and_sparsity():
    ratings_df, _ = create_sample_dataset()
    train_df, _, _ = split_ratings_data(ratings_df)
    matrix = build_interaction_matrix(train_df)

    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0
    sparsity = calculate_sparsity(matrix)
    assert 0.0 <= sparsity <= 1.0


def test_get_movie_metadata_maps():
    _, movies_df = create_sample_dataset()
    id_to_title, title_to_id, genres = get_movie_metadata_maps(movies_df)

    assert len(id_to_title) == len(movies_df)
    assert len(title_to_id) == len(movies_df)
    assert "Comedy" in genres
    assert "Action" in genres
