"""
Embedding Model Module.
Manages loading, caching, and generating embeddings via Sentence-BERT (sentence-transformers) with sklearn fallback.
"""

from typing import List, Union
import numpy as np
from config import EMBEDDING_MODEL_NAME
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)

_GLOBAL_MODEL = None


class FallbackVectorizer:
    """Fallback semantic vectorizer using scikit-learn when SBERT model is offline or loading."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        # Pre-fit on common vocabulary
        base_corpus = [
            "python java c++ machine learning deep learning artificial intelligence",
            "natural language processing computer vision docker kubernetes aws cloud",
            "react nodejs sql postgresql mongodb rest api microservices data science",
            "data analysis pandas numpy scikit-learn tensorflow pytorch devops git",
            "software engineer developer backend frontend fullstack algorithms",
        ]
        self.vectorizer.fit(base_corpus)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generates normalized dense embedding vectors."""
        if not texts:
            return np.zeros((0, 100))
        # Ensure texts are strings
        str_texts = [str(t) if t else "" for t in texts]
        tfidf = self.vectorizer.transform(str_texts).toarray()
        # L2 normalize
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return tfidf / norms


class EmbeddingModel:
    """Encapsulates embedding generation with singleton caching."""

    _instance = None
    _sbert_model = None
    _fallback_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Loads SentenceTransformer model or initializes fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading Sentence-BERT model: %s", EMBEDDING_MODEL_NAME)
            self._sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Sentence-BERT model loaded successfully.")
        except Exception as e:
            logger.warning(
                "Could not load sentence-transformers (%s). Using scikit-learn embedding backend.",
                str(e),
            )
            self._sbert_model = None
            self._fallback_model = FallbackVectorizer()

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encodes a single text or list of texts into dense numpy vector embeddings.

        Args:
            texts: Single string or list of text strings.

        Returns:
            2D numpy array of shape (num_texts, embedding_dim).
        """
        if isinstance(texts, str):
            input_texts = [texts]
        else:
            input_texts = list(texts)

        if not input_texts:
            return np.zeros((0, 384))

        if self._sbert_model is not None:
            try:
                embeddings = self._sbert_model.encode(
                    input_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return np.array(embeddings)
            except Exception as e:
                logger.error("SBERT encoding error: %s, using fallback.", str(e))

        if self._fallback_model is None:
            self._fallback_model = FallbackVectorizer()

        return self._fallback_model.encode(input_texts)
