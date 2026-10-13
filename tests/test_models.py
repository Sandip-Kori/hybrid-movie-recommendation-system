"""Unit tests for ItemBasedCF, KNNRecommender, and HybridRecommender."""

import numpy as np
import pandas as pd
import pytest

from src.config import MAX_RATING, MIN_RATING, PAPER_REFERENCE_ALPHA
from src.data_loader import create_sample_dataset, split_ratings_data
from src.models import HybridRecommender, ItemBasedCF, KNNRecommender


@pytest.fixture
def sample_data():
    ratings_df, movies_df = create_sample_dataset()
    train_df, val_df, test_df = split_ratings_data(ratings_df, chronological=False)
    return train_df, val_df, test_df, movies_df


def test_cf_model_fit_and_predict(sample_data):
    train_df, _, _, _ = sample_data
    cf = ItemBasedCF(k_neighbors=5)
    cf.fit(train_df)

    assert cf.is_fitted
    assert cf.fit_time_seconds > 0.0

    user_id = train_df.iloc[0]["userId"]
    movie_id = train_df.iloc[0]["movieId"]

    pred = cf.predict_rating(user_id, movie_id)
    assert MIN_RATING <= pred <= MAX_RATING


def test_cf_fallback_for_unseen_entities(sample_data):
    train_df, _, _, _ = sample_data
    cf = ItemBasedCF(k_neighbors=5)
    cf.fit(train_df)

    # Unseen user
    pred_unseen_user = cf.predict_rating(user_id=999999, movie_id=train_df.iloc[0]["movieId"])
    assert MIN_RATING <= pred_unseen_user <= MAX_RATING

    # Unseen movie
    pred_unseen_movie = cf.predict_rating(user_id=train_df.iloc[0]["userId"], movie_id=999999)
    assert MIN_RATING <= pred_unseen_movie <= MAX_RATING

    # Both unseen
    pred_both_unseen = cf.predict_rating(user_id=999999, movie_id=999999)
    assert pred_both_unseen == cf.global_mean


def test_knn_model_fit_and_predict(sample_data):
    train_df, _, _, _ = sample_data
    knn = KNNRecommender(k_neighbors=5)
    knn.fit(train_df)

    assert knn.is_fitted
    assert knn.fit_time_seconds > 0.0

    user_id = train_df.iloc[0]["userId"]
    movie_id = train_df.iloc[0]["movieId"]

    pred = knn.predict_rating(user_id, movie_id)
    assert MIN_RATING <= pred <= MAX_RATING


def test_hybrid_formula_exactness(sample_data):
    train_df, _, _, _ = sample_data

    cf = ItemBasedCF(k_neighbors=5).fit(train_df)
    knn = KNNRecommender(k_neighbors=5).fit(train_df)

    user_id = train_df.iloc[0]["userId"]
    movie_id = train_df.iloc[0]["movieId"]

    cf_pred = cf.predict_rating(user_id, movie_id)
    knn_pred = knn.predict_rating(user_id, movie_id)

    # Test arbitrary alpha
    alpha = 0.70
    hybrid = HybridRecommender(alpha=alpha, k_neighbors=5)
    hybrid.fit(train_df)

    hybrid_pred = hybrid.predict_rating(user_id, movie_id)
    expected = alpha * cf_pred + (1.0 - alpha) * knn_pred
    expected_clipped = float(np.clip(expected, MIN_RATING, MAX_RATING))

    assert np.isclose(hybrid_pred, expected_clipped, atol=1e-5)


def test_hybrid_alpha_boundary_values(sample_data):
    train_df, _, _, _ = sample_data

    user_id = train_df.iloc[0]["userId"]
    movie_id = train_df.iloc[0]["movieId"]

    hybrid = HybridRecommender(alpha=1.0, k_neighbors=5).fit(train_df)
    cf = ItemBasedCF(k_neighbors=5).fit(train_df)
    assert np.isclose(hybrid.predict_rating(user_id, movie_id), cf.predict_rating(user_id, movie_id))

    hybrid.set_alpha(0.0)
    knn = KNNRecommender(k_neighbors=5).fit(train_df)
    assert np.isclose(hybrid.predict_rating(user_id, movie_id), knn.predict_rating(user_id, movie_id))


def test_hybrid_rating_clipping(sample_data):
    train_df, _, _, _ = sample_data
    hybrid = HybridRecommender(alpha=0.95, k_neighbors=5).fit(train_df)

    # Mock component predictions that exceed range
    hybrid.cf_model.predict_rating = lambda u, m: 10.0
    hybrid.knn_model.predict_rating = lambda u, m: 10.0
    assert hybrid.predict_rating(1, 1) == MAX_RATING

    hybrid.cf_model.predict_rating = lambda u, m: -5.0
    hybrid.knn_model.predict_rating = lambda u, m: -2.0
    assert hybrid.predict_rating(1, 1) == MIN_RATING


def test_recommend_for_movie_excludes_query(sample_data):
    train_df, _, _, movies_df = sample_data
    hybrid = HybridRecommender(alpha=PAPER_REFERENCE_ALPHA, k_neighbors=5).fit(train_df)

    query_movie_id = int(train_df.iloc[0]["movieId"])
    top_5 = hybrid.recommend_for_movie(query_movie_id, top_n=5, movies_df=movies_df)

    assert len(top_5) <= 5
    for rec in top_5:
        # Excludes query movie
        assert rec["movieId"] != query_movie_id
        assert "title" in rec
        assert "similarity" in rec
        assert "predicted_rating" in rec


def test_recommend_for_movie_genre_filtering(sample_data):
    train_df, _, _, movies_df = sample_data
    hybrid = HybridRecommender(alpha=PAPER_REFERENCE_ALPHA, k_neighbors=5).fit(train_df)

    query_movie_id = int(train_df.iloc[0]["movieId"])
    recs = hybrid.recommend_for_movie(
        query_movie_id, top_n=10, genre_filter="Animation", movies_df=movies_df
    )

    for rec in recs:
        assert "Animation" in rec["genres"]
