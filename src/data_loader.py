"""Dataset loading, preprocessing, and splitting for the Movie Recommendation System."""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.config import (
    DATA_DIR,
    MAX_RATING,
    MIN_RATING,
    MOVIELENS_URL,
    MOVIES_FILE,
    RANDOM_STATE,
    RATINGS_FILE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def create_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a clean synthetic MovieLens-style dataset for offline/fallback use."""
    np.random.seed(RANDOM_STATE)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample_movies = [
        (1, "Toy Story (1995)", "Animation|Children|Comedy"),
        (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
        (3, "Grumpier Old Men (1995)", "Comedy|Romance"),
        (4, "Waiting to Exhale (1995)", "Comedy|Drama|Romance"),
        (5, "Father of the Bride Part II (1995)", "Comedy"),
        (6, "Heat (1995)", "Action|Crime|Thriller"),
        (7, "Sabrina (1995)", "Comedy|Romance"),
        (8, "Tom and Huck (1995)", "Adventure|Children"),
        (9, "Sudden Death (1995)", "Action"),
        (10, "GoldenEye (1995)", "Action|Adventure|Thriller"),
        (11, "American President, The (1995)", "Comedy|Drama|Romance"),
        (12, "Dracula: Dead and Loving It (1995)", "Comedy|Horror"),
        (13, "Balto (1995)", "Animation|Children"),
        (14, "Nixon (1995)", "Drama"),
        (15, "Cutthroat Island (1995)", "Action|Adventure|Romance"),
        (16, "Casino (1995)", "Crime|Drama"),
        (17, "Sense and Sensibility (1995)", "Drama|Romance"),
        (18, "Four Rooms (1995)", "Comedy"),
        (19, "Ace Ventura: When Nature Calls (1995)", "Comedy"),
        (20, "Money Train (1995)", "Action|Comedy|Crime|Drama|Thriller"),
        (21, "Get Shorty (1995)", "Comedy|Crime|Thriller"),
        (22, "Copycat (1995)", "Crime|Drama|Mystery|Thriller"),
        (23, "Assassins (1995)", "Action|Crime|Thriller"),
        (24, "Powder (1995)", "Drama|Sci-Fi"),
        (25, "Leaving Las Vegas (1995)", "Drama|Romance"),
        (29, "City of Lost Children, The (1995)", "Adventure|Drama|Fantasy|Mystery|Sci-Fi"),
        (31, "Dangerous Minds (1995)", "Drama"),
        (32, "Twelve Monkeys (a.k.a. 12 Monkeys) (1995)", "Mystery|Sci-Fi|Thriller"),
        (34, "Babe (1995)", "Children|Drama"),
        (36, "Dead Man Walking (1995)", "Crime|Drama"),
        (39, "Clueless (1995)", "Comedy|Romance"),
        (47, "Seven (a.k.a. Se7en) (1995)", "Mystery|Thriller"),
        (50, "Usual Suspects, The (1995)", "Crime|Mystery|Thriller"),
        (110, "Braveheart (1995)", "Action|Drama|War"),
        (150, "Apollo 13 (1995)", "Adventure|Drama|IMAX"),
        (260, "Star Wars: Episode IV - A New Hope (1977)", "Action|Adventure|Sci-Fi"),
        (296, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller"),
        (318, "Shawshank Redemption, The (1994)", "Crime|Drama"),
        (356, "Forrest Gump (1994)", "Comedy|Drama|Romance|War"),
        (527, "Schindler's List (1993)", "Drama|War"),
        (589, "Terminator 2: Judgment Day (1991)", "Action|Sci-Fi"),
        (593, "Silence of the Lambs, The (1991)", "Crime|Horror|Thriller"),
        (858, "Godfather, The (1972)", "Crime|Drama"),
        (1196, "Star Wars: Episode V - The Empire Strikes Back (1980)", "Action|Adventure|Sci-Fi"),
        (1210, "Star Wars: Episode VI - Return of the Jedi (1983)", "Action|Adventure|Sci-Fi"),
        (2571, "Matrix, The (1999)", "Action|Sci-Fi|Thriller"),
        (2959, "Fight Club (1999)", "Action|Crime|Drama|Thriller"),
        (4993, "Lord of the Rings: The Fellowship of the Ring, The (2001)", "Adventure|Fantasy"),
        (5952, "Lord of the Rings: The Two Towers, The (2002)", "Adventure|Fantasy"),
        (7153, "Lord of the Rings: The Return of the King, The (2003)", "Action|Adventure|Drama|Fantasy"),
    ]

    movies_df = pd.DataFrame(sample_movies, columns=["movieId", "title", "genres"])

    # Generate realistic rating interactions
    n_users = 120
    ratings_records = []
    base_timestamp = 1000000000

    movie_ids = [m[0] for m in sample_movies]
    for user_id in range(1, n_users + 1):
        # Each user rates between 15 and 35 movies
        n_ratings = np.random.randint(15, 36)
        user_movies = np.random.choice(movie_ids, size=n_ratings, replace=False)
        user_bias = np.random.normal(3.5, 0.4)
        for idx, m_id in enumerate(user_movies):
            # Rating centered around user bias with random noise, clipped to [1.0, 5.0]
            raw_rating = np.round(np.random.normal(user_bias, 0.7) * 2) / 2
            rating = max(MIN_RATING, min(MAX_RATING, raw_rating))
            timestamp = base_timestamp + user_id * 1000 + idx * 50
            ratings_records.append((user_id, m_id, rating, timestamp))

    ratings_df = pd.DataFrame(
        ratings_records, columns=["userId", "movieId", "rating", "timestamp"]
    )

    movies_df.to_csv(MOVIES_FILE, index=False)
    ratings_df.to_csv(RATINGS_FILE, index=False)

    return ratings_df, movies_df


def download_movielens_dataset(force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download MovieLens dataset or fallback to sample generator if offline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force_download and RATINGS_FILE.exists() and MOVIES_FILE.exists():
        ratings_df = pd.read_csv(RATINGS_FILE)
        movies_df = pd.read_csv(MOVIES_FILE)
        return ratings_df, movies_df

    try:
        response = requests.get(MOVIELENS_URL, timeout=12)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Files inside zip are typically under ml-latest-small/
            for filename in z.namelist():
                if filename.endswith("ratings.csv"):
                    with z.open(filename) as f:
                        ratings_df = pd.read_csv(f)
                        ratings_df.to_csv(RATINGS_FILE, index=False)
                elif filename.endswith("movies.csv"):
                    with z.open(filename) as f:
                        movies_df = pd.read_csv(f)
                        movies_df.to_csv(MOVIES_FILE, index=False)

        if RATINGS_FILE.exists() and MOVIES_FILE.exists():
            return pd.read_csv(RATINGS_FILE), pd.read_csv(MOVIES_FILE)

    except Exception:
        # Graceful offline fallback
        pass

    return create_sample_dataset()


def clean_and_validate_data(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and validate data: remove duplicates, nulls, and out-of-range ratings."""
    clean_ratings = ratings_df.dropna(subset=["userId", "movieId", "rating"]).copy()
    clean_ratings["userId"] = clean_ratings["userId"].astype(int)
    clean_ratings["movieId"] = clean_ratings["movieId"].astype(int)
    clean_ratings["rating"] = clean_ratings["rating"].astype(float)

    # Filter valid rating range
    clean_ratings = clean_ratings[
        (clean_ratings["rating"] >= MIN_RATING)
        & (clean_ratings["rating"] <= MAX_RATING)
    ]
    clean_ratings = clean_ratings.drop_duplicates(subset=["userId", "movieId"])

    clean_movies = movies_df.dropna(subset=["movieId", "title"]).copy()
    clean_movies["movieId"] = clean_movies["movieId"].astype(int)
    clean_movies = clean_movies.drop_duplicates(subset=["movieId"])

    # Ensure movie IDs in ratings match known movies
    known_movies = set(clean_movies["movieId"].unique())
    clean_ratings = clean_ratings[clean_ratings["movieId"].isin(known_movies)]

    return clean_ratings.reset_index(drop=True), clean_movies.reset_index(drop=True)


def split_ratings_data(
    ratings_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    chronological: bool = True,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into Train, Validation, and Test sets.
    Ensures validation and test splits are strictly separated to prevent leakage.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df = ratings_df.copy()
    if chronological and "timestamp" in df.columns:
        df = df.sort_values(by="timestamp").reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def build_interaction_matrix(train_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build user-item interaction matrix from training split.
    Rows: userId, Columns: movieId.
    """
    matrix = train_ratings.pivot(index="userId", columns="movieId", values="rating")
    return matrix


def calculate_sparsity(interaction_matrix: pd.DataFrame) -> float:
    """Calculate the sparsity ratio of the user-item interaction matrix."""
    total_cells = interaction_matrix.shape[0] * interaction_matrix.shape[1]
    if total_cells == 0:
        return 0.0
    rated_cells = interaction_matrix.notna().sum().sum()
    sparsity = 1.0 - (rated_cells / total_cells)
    return float(sparsity)


def get_movie_metadata_maps(
    movies_df: pd.DataFrame,
) -> Tuple[Dict[int, str], Dict[str, int], List[str]]:
    """Build fast lookup dictionaries for movie titles, IDs, and list of all genres."""
    id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    title_to_id = dict(zip(movies_df["title"], movies_df["movieId"]))

    genres_set = set()
    for genre_str in movies_df["genres"].dropna():
        for g in genre_str.split("|"):
            if g and g != "(no genres listed)":
                genres_set.add(g)

    return id_to_title, title_to_id, sorted(list(genres_set))
