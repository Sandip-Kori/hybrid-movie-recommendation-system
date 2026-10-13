"""Unit tests for model evaluation and alpha tuning."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.data_loader import create_sample_dataset, split_ratings_data
from src.evaluation import (
    calculate_mae,
    calculate_rmse,
    compare_all_models,
    evaluate_model,
    generate_comparison_chart,
    tune_alpha_on_validation,
)
from src.models import ItemBasedCF, KNNRecommender


def test_rmse_and_mae_calculation():
    y_true = np.array([3.0, 4.0, 5.0, 2.0])
    y_pred = np.array([3.0, 4.0, 5.0, 2.0])
    assert calculate_rmse(y_true, y_pred) == 0.0
    assert calculate_mae(y_true, y_pred) == 0.0

    y_pred_diff = np.array([4.0, 3.0, 4.0, 3.0])
    # Differences: [1, -1, -1, 1], squared: [1, 1, 1, 1], mean: 1, sqrt: 1
    assert np.isclose(calculate_rmse(y_true, y_pred_diff), 1.0)
    assert np.isclose(calculate_mae(y_true, y_pred_diff), 1.0)


def test_evaluate_model_returns_metrics():
    ratings_df, _ = create_sample_dataset()
    train_df, _, test_df = split_ratings_data(ratings_df, chronological=False)

    cf = ItemBasedCF(k_neighbors=5).fit(train_df)
    metrics = evaluate_model(cf, test_df)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "fit_time_seconds" in metrics
    assert "eval_time_seconds" in metrics
    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    assert metrics["n_eval_samples"] == len(test_df)


def test_tune_alpha_on_validation():
    ratings_df, _ = create_sample_dataset()
    train_df, val_df, _ = split_ratings_data(ratings_df, chronological=False)

    cf = ItemBasedCF(k_neighbors=5).fit(train_df)
    knn = KNNRecommender(k_neighbors=5).fit(train_df)

    candidates = [0.10, 0.50, 0.95]
    best_alpha, history_df = tune_alpha_on_validation(cf, knn, val_df, candidate_alphas=candidates)

    assert best_alpha in candidates
    assert len(history_df) == len(candidates)
    assert "val_rmse" in history_df.columns
    assert "val_mae" in history_df.columns


def test_compare_all_models_and_chart():
    ratings_df, _ = create_sample_dataset()
    train_df, val_df, test_df = split_ratings_data(ratings_df, chronological=False)

    results_df, best_alpha, val_history, hybrid_model = compare_all_models(
        train_df, val_df, test_df, k_neighbors=5
    )

    assert len(results_df) >= 3
    assert "RMSE" in results_df.columns
    assert "MAE" in results_df.columns
    assert "Train Time (s)" in results_df.columns

    fig = generate_comparison_chart(results_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
