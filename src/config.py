"""Configuration and constants for the Hybrid Movie Recommendation System."""

from pathlib import Path
from typing import List

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"

# Dataset URL (MovieLens Latest Small: ~100k ratings, 9k movies)
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Reproducibility
RANDOM_STATE = 42

# Rating bounds
MIN_RATING = 1.0
MAX_RATING = 5.0

# Split ratios (chronological or seeded train/val/test)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Algorithm Hyperparameters
DEFAULT_K_NEIGHBORS = 20  # k=20 from the research paper
PAPER_REFERENCE_ALPHA = 0.95  # Converged reference alpha reported in the paper

# Alpha candidate grid for validation tuning (from Paper Section III-C)
GRID_ALPHAS: List[float] = [
    round(a, 2) for a in [0.10 + i * 0.05 for i in range(18)]
]

# UI styling constants
THEME_NAVY = "#0f172a"
THEME_CARD_BG = "#1e293b"
THEME_ACCENT_PURPLE = "#8b5cf6"
THEME_ACCENT_BLUE = "#3b82f6"
THEME_TEXT = "#f8fafc"
