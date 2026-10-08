"""Semantic matching package."""
from .embedding_model import EmbeddingModel
from .similarity import compute_cosine_similarity, batch_cosine_similarity
from .matcher import SemanticSkillMatcher

__all__ = [
    "EmbeddingModel",
    "compute_cosine_similarity",
    "batch_cosine_similarity",
    "SemanticSkillMatcher",
]
