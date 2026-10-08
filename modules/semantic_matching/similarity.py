"""
Similarity Module.
Calculates cosine similarity metrics between embedding vectors using scikit-learn.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D or 2D embedding vectors.

    Args:
        vec1: First embedding vector (1D or 2D).
        vec2: Second embedding vector (1D or 2D).

    Returns:
        Cosine similarity score as float between 0.0 and 1.0 (clamped).
    """
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return 0.0

    v1 = np.array(vec1).reshape(1, -1) if np.array(vec1).ndim == 1 else np.array(vec1)
    v2 = np.array(vec2).reshape(1, -1) if np.array(vec2).ndim == 1 else np.array(vec2)

    sim_matrix = cosine_similarity(v1, v2)
    score = float(sim_matrix[0][0])
    # Clamp between 0.0 and 1.0
    return max(0.0, min(1.0, score))


def batch_cosine_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity matrix between two batches of embeddings.

    Args:
        matrix1: 2D array of shape (N, D).
        matrix2: 2D array of shape (M, D).

    Returns:
        2D array of shape (N, M) with values clamped between 0.0 and 1.0.
    """
    if len(matrix1) == 0 or len(matrix2) == 0:
        return np.zeros((len(matrix1), len(matrix2)))

    sim = cosine_similarity(matrix1, matrix2)
    return np.clip(sim, 0.0, 1.0)
