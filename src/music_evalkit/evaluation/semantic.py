import numpy as np


class SemanticSimilarity:
    """Compute semantic similarity between texts using SBERT.

    Uses sentence-transformers library with all-MiniLM-L6-v2 model.
    Runs locally on CPU, no API needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with SBERT model.

        Args:
            model_name: SBERT model name. Options:
                - 'all-MiniLM-L6-v2' (default, 80MB, fast)
                - 'all-mpnet-base-v2' (420MB, better quality)
                - 'paraphrase-MiniLM-L6-v2' (good for paraphrase)
        """
        self._model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1.
        """
        self._load_model()

        # Encode both texts
        embeddings = self._model.encode([text1, text2])

        # Compute cosine similarity
        return self._cosine_similarity(embeddings[0], embeddings[1])

    def similarity_batch(
        self, predictions: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Compute similarity for multiple pairs.

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts (same length)

        Returns:
            List of similarity scores.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have same length")

        self._load_model()

        # Encode all texts at once (more efficient)
        pred_embeddings = self._model.encode(predictions)
        truth_embeddings = self._model.encode(ground_truths)

        # Compute pairwise cosine similarity
        scores = []
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
            scores.append(self._cosine_similarity(pred_emb, truth_emb))

        return scores

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def similarity_with_null_handling(
        self, pred: str | None, truth: str | None
    ) -> tuple[float, str]:
        """Compute similarity with special handling for null values.

        Args:
            pred: Predicted text (can be None for no_mistake)
            truth: Ground truth text (can be None for no_mistake)

        Returns:
            Tuple of (similarity_score, match_type) where match_type is:
            - 'both_null': Both are None (correct no_mistake)
            - 'pred_null': Only prediction is None
            - 'truth_null': Only ground truth is None
            - 'compared': Both have text, similarity computed
        """
        if pred is None and truth is None:
            return 1.0, "both_null"
        elif pred is None:
            return 0.0, "pred_null"
        elif truth is None:
            return 0.0, "truth_null"
        else:
            return self.similarity(pred, truth), "compared"
