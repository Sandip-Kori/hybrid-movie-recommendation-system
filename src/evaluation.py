"""Model evaluation, alpha tuning on validation data, and comparison metrics."""

import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers/UI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import GRID_ALPHAS, PAPER_REFERENCE_ALPHA
from src.models import BaseRecommender, HybridRecommender, ItemBasedCF, KNNRecommender


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_model(
    model: BaseRecommender, eval_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate a fitted recommender model on a given evaluation split.
    Returns RMSE, MAE, evaluation time, and fitting time.
    """
    start_time = time.perf_counter()

    y_true = []
    y_pred = []

    for _, row in eval_df.iterrows():
        u_id = int(row["userId"])
        m_id = int(row["movieId"])
        true_r = float(row["rating"])

        pred_r = model.predict_rating(u_id, m_id)

        y_true.append(true_r)
        y_pred.append(pred_r)

    eval_time = time.perf_counter() - start_time

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    rmse = calculate_rmse(y_true_arr, y_pred_arr)
    mae = calculate_mae(y_true_arr, y_pred_arr)

    return {
        "rmse": float(round(rmse, 4)),
        "mae": float(round(mae, 4)),
        "fit_time_seconds": float(round(model.fit_time_seconds, 4)),
        "eval_time_seconds": float(round(eval_time, 4)),
        "n_eval_samples": int(len(eval_df)),
    }


def tune_alpha_on_validation(
    cf_model: ItemBasedCF,
    knn_model: KNNRecommender,
    val_df: pd.DataFrame,
    candidate_alphas: Optional[List[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Tune hybrid blending weight alpha strictly on validation split.
    Tests candidate alphas and selects the one that minimizes RMSE.
    """
    if candidate_alphas is None:
        candidate_alphas = GRID_ALPHAS

    # Precompute CF and KNN predictions on validation set for rapid alpha search
    cf_preds = []
    knn_preds = []
    y_true = []

    for _, row in val_df.iterrows():
        u_id = int(row["userId"])
        m_id = int(row["movieId"])
        y_true.append(float(row["rating"]))
        cf_preds.append(cf_model.predict_rating(u_id, m_id))
        knn_preds.append(knn_model.predict_rating(u_id, m_id))

    cf_arr = np.array(cf_preds)
    knn_arr = np.array(knn_preds)
    true_arr = np.array(y_true)

    history = []
    best_alpha = PAPER_REFERENCE_ALPHA
    best_rmse = float("inf")

    for a in candidate_alphas:
        # hybrid = a * cf + (1 - a) * knn
        blended = np.clip(a * cf_arr + (1.0 - a) * knn_arr, 1.0, 5.0)
        curr_rmse = calculate_rmse(true_arr, blended)
        curr_mae = calculate_mae(true_arr, blended)

        history.append({"alpha": round(a, 2), "val_rmse": round(curr_rmse, 4), "val_mae": round(curr_mae, 4)})

        if curr_rmse < best_rmse:
            best_rmse = curr_rmse
            best_alpha = round(a, 2)

    history_df = pd.DataFrame(history)
    return best_alpha, history_df


def compare_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_neighbors: int = 20,
) -> Tuple[pd.DataFrame, float, pd.DataFrame, HybridRecommender]:
    """
    Full workflow:
    1. Fit ItemBasedCF and KNN on train split.
    2. Tune alpha on validation split.
    3. Fit Hybrid with tuned alpha and paper reference alpha.
    4. Evaluate all models on held-out test split.
    """
    # 1. Fit individual models
    cf = ItemBasedCF(k_neighbors=k_neighbors).fit(train_df)
    knn = KNNRecommender(k_neighbors=k_neighbors).fit(train_df)

    # 2. Validation alpha tuning
    best_alpha, val_tuning_history = tune_alpha_on_validation(cf, knn, val_df)

    # 3. Create and fit hybrid models
    hybrid_tuned = HybridRecommender(alpha=best_alpha, k_neighbors=k_neighbors)
    hybrid_tuned.fit(train_df)

    # 4. Evaluate all on unseen test split
    cf_metrics = evaluate_model(cf, test_df)
    knn_metrics = evaluate_model(knn, test_df)
    hybrid_metrics = evaluate_model(hybrid_tuned, test_df)

    records = [
        {
            "Model": "Item-Based CF",
            "RMSE": cf_metrics["rmse"],
            "MAE": cf_metrics["mae"],
            "Train Time (s)": cf_metrics["fit_time_seconds"],
            "Eval Time (s)": cf_metrics["eval_time_seconds"],
            "Configuration": f"k={k_neighbors}",
        },
        {
            "Model": "Standalone KNN",
            "RMSE": knn_metrics["rmse"],
            "MAE": knn_metrics["mae"],
            "Train Time (s)": knn_metrics["fit_time_seconds"],
            "Eval Time (s)": knn_metrics["eval_time_seconds"],
            "Configuration": f"k={k_neighbors}, metric=cosine",
        },
        {
            "Model": f"Hybrid CF+KNN (alpha={best_alpha})",
            "RMSE": hybrid_metrics["rmse"],
            "MAE": hybrid_metrics["mae"],
            "Train Time (s)": hybrid_metrics["fit_time_seconds"],
            "Eval Time (s)": hybrid_metrics["eval_time_seconds"],
            "Configuration": f"alpha={best_alpha} (Tuned on Val)",
        },
    ]

    # If tuned alpha differs from paper reference alpha 0.95, also evaluate reference hybrid
    if best_alpha != PAPER_REFERENCE_ALPHA:
        hybrid_ref = HybridRecommender(alpha=PAPER_REFERENCE_ALPHA, k_neighbors=k_neighbors)
        hybrid_ref.fit(train_df)
        ref_metrics = evaluate_model(hybrid_ref, test_df)
        records.append(
            {
                "Model": f"Hybrid CF+KNN (alpha={PAPER_REFERENCE_ALPHA} Paper Ref)",
                "RMSE": ref_metrics["rmse"],
                "MAE": ref_metrics["mae"],
                "Train Time (s)": ref_metrics["fit_time_seconds"],
                "Eval Time (s)": ref_metrics["eval_time_seconds"],
                "Configuration": f"alpha={PAPER_REFERENCE_ALPHA} (Paper Ref)",
            }
        )

    results_df = pd.DataFrame(records)
    return results_df, best_alpha, val_tuning_history, hybrid_tuned


def generate_comparison_chart(results_df: pd.DataFrame) -> plt.Figure:
    """Generate a clean Matplotlib comparison chart for RMSE, MAE, and Training Time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#0f172a")

    models = results_df["Model"].tolist()
    rmse_vals = results_df["RMSE"].tolist()
    mae_vals = results_df["MAE"].tolist()
    time_vals = results_df["Train Time (s)"].tolist()

    x = np.arange(len(models))
    width = 0.35

    # Panel 1: RMSE & MAE
    ax1.set_facecolor("#1e293b")
    rects1 = ax1.bar(x - width / 2, rmse_vals, width, label="RMSE", color="#6366f1")
    rects2 = ax1.bar(x + width / 2, mae_vals, width, label="MAE", color="#8b5cf6")

    ax1.set_title("Test Accuracy Comparison", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", color="#cbd5e1", fontsize=9)
    ax1.set_ylabel("Error (Lower is better)", color="#cbd5e1")
    ax1.tick_params(colors="#cbd5e1")
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")

    # Add bar values
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", color="#e2e8f0", fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    # Panel 2: Training Execution Time
    ax2.set_facecolor("#1e293b")
    time_bars = ax2.bar(x, time_vals, width=0.5, color="#38bdf8", label="Train Time (s)")
    ax2.set_title("Training Execution Time (s)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", color="#cbd5e1", fontsize=9)
    ax2.set_ylabel("Seconds", color="#cbd5e1")
    ax2.tick_params(colors="#cbd5e1")
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")

    for r in time_bars:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}s", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    plt.tight_layout()
    return fig
