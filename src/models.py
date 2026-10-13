"""Recommendation algorithms: Item-Based CF, KNN, and Alpha-Weighted Hybrid."""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.config import (
    DEFAULT_K_NEIGHBORS,
    MAX_RATING,
    MIN_RATING,
    PAPER_REFERENCE_ALPHA,
)


class BaseRecommender:
    """Abstract base class for all recommender implementations."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.fit_time_seconds: float = 0.0
        self.global_mean: float = 3.5
        self.user_means: Dict[int, float] = {}
        self.item_means: Dict[int, float] = {}

    def fit(self, train_ratings: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        raise NotImplementedError

    def recommend_for_movie(
        self,
        movie_id: int,
        top_n: int = 5,
        genre_filter: Optional[str] = None,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Union[int, str, float]]]:
        raise NotImplementedError


class ItemBasedCF(BaseRecommender):
    """
    Item-Based Collaborative Filtering Recommender.
    Computes pairwise item cosine similarities and estimates ratings
    using the top-k most similar items rated by the target user.
    """

    def __init__(self, k_neighbors: int = DEFAULT_K_NEIGHBORS):
        super().__init__(name="Item-Based CF")
        self.k_neighbors = k_neighbors
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_df: Optional[pd.DataFrame] = None
        self.movie_ids: List[int] = []

    def fit(self, train_ratings: pd.DataFrame) -> "ItemBasedCF":
        start_time = time.perf_counter()

        # Compute global and group means for fallbacks
        self.global_mean = float(train_ratings["rating"].mean())
        self.user_means = train_ratings.groupby("userId")["rating"].mean().to_dict()
        self.item_means = train_ratings.groupby("movieId")["rating"].mean().to_dict()

        # Pivot user-item interaction matrix
        pivot = train_ratings.pivot(index="userId", columns="movieId", values="rating")
        self.user_item_matrix = pivot
        self.movie_ids = list(pivot.columns)

        # Center ratings per user to compute adjusted cosine similarity (or mean-imputed)
        user_mean_series = pivot.mean(axis=1)
        matrix_centered = pivot.sub(user_mean_series, axis=0).fillna(0.0)

        # Compute pairwise item cosine similarity
        item_sim = cosine_similarity(matrix_centered.T)
        np.fill_diagonal(item_sim, 0.0)  # item similarity to itself is not considered a neighbor

        self.item_similarity_df = pd.DataFrame(
            item_sim, index=self.movie_ids, columns=self.movie_ids
        )

        self.fit_time_seconds = time.perf_counter() - start_time
        self.is_fitted = True
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using item-based collaborative filtering formula with documented fallbacks:
        1. Item mean rating if no rated neighbors.
        2. User mean rating if item has no training ratings.
        3. Global mean if user and item are unseen.
        """
        # Fallback Level 3: Unseen movie
        if movie_id not in self.item_means:
            return float(self.user_means.get(user_id, self.global_mean))

        # Fallback Level 2: Unseen user
        if user_id not in self.user_means:
            return float(self.item_means.get(movie_id, self.global_mean))

        if self.user_item_matrix is None or self.item_similarity_df is None:
            return float(self.global_mean)

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        if user_ratings.empty:
            return float(self.item_means.get(movie_id, self.user_means.get(user_id, self.global_mean)))

        # Find similarities between candidate movie and items rated by user
        sims = self.item_similarity_df.loc[movie_id, user_ratings.index]
        positive_sims = sims[sims > 0]

        if positive_sims.empty:
            # Fallback Level 1: No positive similar items rated by this user
            return float(self.item_means.get(movie_id, self.user_means.get(user_id, self.global_mean)))

        top_k_sims = positive_sims.nlargest(self.k_neighbors)
        top_k_ratings = user_ratings[top_k_sims.index]

        numerator = np.sum(top_k_sims.values * top_k_ratings.values)
        denominator = np.sum(np.abs(top_k_sims.values))

        if denominator == 0:
            return float(self.item_means.get(movie_id, self.global_mean))

        prediction = numerator / denominator
        return float(np.clip(prediction, MIN_RATING, MAX_RATING))

    def recommend_for_movie(
        self,
        movie_id: int,
        top_n: int = 5,
        genre_filter: Optional[str] = None,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Union[int, str, float]]]:
        """Rank most similar movies to query movie, excluding the movie itself."""
        if not self.is_fitted or self.item_similarity_df is None:
            return []

        if movie_id not in self.item_similarity_df.index:
            return []

        sim_scores = self.item_similarity_df.loc[movie_id].copy()
        sim_scores = sim_scores.drop(index=movie_id, errors="ignore")

        # Map metadata if provided
        results = []
        movies_lookup = {}
        if movies_df is not None:
            for _, row in movies_df.iterrows():
                movies_lookup[int(row["movieId"])] = {
                    "title": str(row["title"]),
                    "genres": str(row.get("genres", "")),
                }

        sorted_candidates = sim_scores.sort_values(ascending=False)

        count = 0
        for candidate_id, score in sorted_candidates.items():
            candidate_id = int(candidate_id)
            movie_meta = movies_lookup.get(candidate_id, {"title": f"Movie {candidate_id}", "genres": ""})
            genres_str = movie_meta["genres"]

            if genre_filter and genre_filter.lower() != "all":
                if genre_filter.lower() not in genres_str.lower():
                    continue

            results.append(
                {
                    "movieId": candidate_id,
                    "title": movie_meta["title"],
                    "genres": genres_str,
                    "similarity": float(round(score, 4)),
                    "predicted_rating": float(round(self.item_means.get(candidate_id, self.global_mean), 2)),
                }
            )
            count += 1
            if count >= top_n:
                break

        return results


class KNNRecommender(BaseRecommender):
    """
    K-Nearest Neighbours Recommender.
    Uses NearestNeighbors with cosine metric to predict rating regression
    and provide stable predictions in sparse regions.
    """

    def __init__(self, k_neighbors: int = DEFAULT_K_NEIGHBORS):
        super().__init__(name="KNN Recommender")
        self.k_neighbors = k_neighbors
        self.model: Optional[NearestNeighbors] = None
        self.item_features: Optional[np.ndarray] = None
        self.movie_ids: List[int] = []
        self.movie_id_to_idx: Dict[int, int] = {}
        self.user_item_matrix: Optional[pd.DataFrame] = None

    def fit(self, train_ratings: pd.DataFrame) -> "KNNRecommender":
        start_time = time.perf_counter()

        self.global_mean = float(train_ratings["rating"].mean())
        self.user_means = train_ratings.groupby("userId")["rating"].mean().to_dict()
        self.item_means = train_ratings.groupby("movieId")["rating"].mean().to_dict()

        pivot = train_ratings.pivot(index="userId", columns="movieId", values="rating")
        self.user_item_matrix = pivot
        self.movie_ids = list(pivot.columns)
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}

        # Fill NaNs with item mean (or 0) for feature space representation
        filled_matrix = pivot.fillna(0.0).T.values  # shape: (n_items, n_users)
        self.item_features = filled_matrix

        # Fit Scikit-Learn NearestNeighbors
        k = min(self.k_neighbors + 1, len(self.movie_ids))
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        self.model.fit(self.item_features)

        self.fit_time_seconds = time.perf_counter() - start_time
        self.is_fitted = True
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using KNN item neighborhood.
        Falls back to item mean -> user mean -> global mean.
        """
        if movie_id not in self.movie_id_to_idx:
            return float(self.user_means.get(user_id, self.global_mean))
        if user_id not in self.user_means:
            return float(self.item_means.get(movie_id, self.global_mean))

        if self.model is None or self.item_features is None or self.user_item_matrix is None:
            return float(self.global_mean)

        target_idx = self.movie_id_to_idx[movie_id]
        feature_vec = self.item_features[target_idx].reshape(1, -1)

        distances, indices = self.model.kneighbors(feature_vec)
        # distances is cosine distance in [0, 2], similarity = 1 - distance
        neighbor_indices = indices[0][1:]  # exclude self
        neighbor_distances = distances[0][1:]

        user_ratings = self.user_item_matrix.loc[user_id]
        valid_ratings = []
        valid_weights = []

        for n_idx, dist in zip(neighbor_indices, neighbor_distances):
            n_mid = self.movie_ids[n_idx]
            rating = user_ratings.get(n_mid, np.nan)
            if not np.isnan(rating):
                sim = max(0.0, 1.0 - dist)
                valid_ratings.append(rating)
                valid_weights.append(sim)

        if not valid_ratings or np.sum(valid_weights) == 0:
            return float(self.item_means.get(movie_id, self.user_means.get(user_id, self.global_mean)))

        prediction = np.sum(np.array(valid_ratings) * np.array(valid_weights)) / np.sum(valid_weights)
        return float(np.clip(prediction, MIN_RATING, MAX_RATING))

    def recommend_for_movie(
        self,
        movie_id: int,
        top_n: int = 5,
        genre_filter: Optional[str] = None,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Union[int, str, float]]]:
        """Find nearest neighbors for query movie using fitted model."""
        if not self.is_fitted or self.model is None or self.item_features is None:
            return []

        if movie_id not in self.movie_id_to_idx:
            return []

        target_idx = self.movie_id_to_idx[movie_id]
        feature_vec = self.item_features[target_idx].reshape(1, -1)

        # Retrieve more neighbors to allow for genre filtering
        n_candidates = min(len(self.movie_ids), max(top_n * 3, 20))
        distances, indices = self.model.kneighbors(feature_vec, n_neighbors=n_candidates)

        results = []
        movies_lookup = {}
        if movies_df is not None:
            for _, row in movies_df.iterrows():
                movies_lookup[int(row["movieId"])] = {
                    "title": str(row["title"]),
                    "genres": str(row.get("genres", "")),
                }

        count = 0
        for n_idx, dist in zip(indices[0], distances[0]):
            candidate_id = self.movie_ids[n_idx]
            if candidate_id == movie_id:
                continue

            movie_meta = movies_lookup.get(candidate_id, {"title": f"Movie {candidate_id}", "genres": ""})
            genres_str = movie_meta["genres"]

            if genre_filter and genre_filter.lower() != "all":
                if genre_filter.lower() not in genres_str.lower():
                    continue

            similarity = max(0.0, 1.0 - dist)
            results.append(
                {
                    "movieId": candidate_id,
                    "title": movie_meta["title"],
                    "genres": genres_str,
                    "similarity": float(round(similarity, 4)),
                    "predicted_rating": float(round(self.item_means.get(candidate_id, self.global_mean), 2)),
                }
            )
            count += 1
            if count >= top_n:
                break

        return results


class HybridRecommender(BaseRecommender):
    """
    Hybrid Recommender combining Item-Based CF and KNN.
    Formula:
        hybrid_prediction = alpha * cf_prediction + (1 - alpha) * knn_prediction
    All predictions are strictly clipped to [MIN_RATING, MAX_RATING].
    """

    def __init__(
        self,
        alpha: float = PAPER_REFERENCE_ALPHA,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
    ):
        super().__init__(name="Hybrid CF+KNN")
        self.alpha = float(alpha)
        self.k_neighbors = k_neighbors
        self.cf_model = ItemBasedCF(k_neighbors=k_neighbors)
        self.knn_model = KNNRecommender(k_neighbors=k_neighbors)

    def set_alpha(self, alpha: float) -> None:
        """Update alpha blending weight (constrained to [0.0, 1.0])."""
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def fit(self, train_ratings: pd.DataFrame) -> "HybridRecommender":
        start_time = time.perf_counter()

        self.cf_model.fit(train_ratings)
        self.knn_model.fit(train_ratings)

        self.global_mean = self.cf_model.global_mean
        self.user_means = self.cf_model.user_means
        self.item_means = self.cf_model.item_means

        self.fit_time_seconds = time.perf_counter() - start_time
        self.is_fitted = True
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Compute alpha-weighted hybrid rating prediction.
        hybrid_prediction = alpha * cf_prediction + (1 - alpha) * knn_prediction
        Clipped to [1.0, 5.0].
        """
        cf_pred = self.cf_model.predict_rating(user_id, movie_id)
        knn_pred = self.knn_model.predict_rating(user_id, movie_id)

        hybrid_pred = (self.alpha * cf_pred) + ((1.0 - self.alpha) * knn_pred)
        return float(np.clip(hybrid_pred, MIN_RATING, MAX_RATING))

    def recommend_for_movie(
        self,
        movie_id: int,
        top_n: int = 5,
        genre_filter: Optional[str] = None,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Rank items for query movie by blending CF and KNN similarities.
        sim_hybrid = alpha * sim_CF + (1 - alpha) * sim_KNN
        """
        if not self.is_fitted:
            return []

        # Get broader candidate lists from both models
        cf_recs = self.cf_model.recommend_for_movie(
            movie_id, top_n=50, genre_filter=None, movies_df=movies_df
        )
        knn_recs = self.knn_model.recommend_for_movie(
            movie_id, top_n=50, genre_filter=None, movies_df=movies_df
        )

        candidates: Dict[int, Dict[str, Union[int, str, float]]] = {}

        for item in cf_recs:
            mid = int(item["movieId"])
            candidates[mid] = {
                "movieId": mid,
                "title": item["title"],
                "genres": item["genres"],
                "cf_sim": float(item["similarity"]),
                "knn_sim": 0.0,
            }

        for item in knn_recs:
            mid = int(item["movieId"])
            if mid in candidates:
                candidates[mid]["knn_sim"] = float(item["similarity"])
            else:
                candidates[mid] = {
                    "movieId": mid,
                    "title": item["title"],
                    "genres": item["genres"],
                    "cf_sim": 0.0,
                    "knn_sim": float(item["similarity"]),
                }

        results = []
        for mid, data in candidates.items():
            genres_str = str(data["genres"])
            if genre_filter and genre_filter.lower() != "all":
                if genre_filter.lower() not in genres_str.lower():
                    continue

            sim_hybrid = (self.alpha * float(data["cf_sim"])) + (
                (1.0 - self.alpha) * float(data["knn_sim"])
            )
            est_rating = float(self.item_means.get(mid, self.global_mean))

            results.append(
                {
                    "movieId": mid,
                    "title": data["title"],
                    "genres": genres_str,
                    "similarity": float(round(sim_hybrid, 4)),
                    "predicted_rating": float(round(est_rating, 2)),
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_n]
