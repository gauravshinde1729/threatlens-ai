"""CVE description classifier using sentence-transformer embeddings + linear head."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CATEGORIES: list[str] = [
    "injection",
    "auth_bypass",
    "memory_corruption",
    "info_disclosure",
    "dos",
    "privilege_escalation",
    "other",
]

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_EPOCHS = 10
_DEFAULT_LR = 0.001


class _LinearHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CVETextClassifier:
    """Classify CVE descriptions into vulnerability categories.

    Embeddings are produced by a sentence-transformers model; a lightweight
    linear layer is trained on top via PyTorch.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._encoder = SentenceTransformer(model_name)
        self._categories = CATEGORIES
        self._label2idx: dict[str, int] = {c: i for i, c in enumerate(self._categories)}
        self._idx2label: dict[int, str] = {i: c for i, c in enumerate(self._categories)}
        self._head: _LinearHead = _LinearHead(_EMBEDDING_DIM, len(self._categories))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._head.to(self._device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        descriptions: list[str],
        labels: list[str],
        epochs: int = _DEFAULT_EPOCHS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        lr: float = _DEFAULT_LR,
    ) -> "CVETextClassifier":
        """Encode descriptions and train the linear classifier head.

        Args:
            descriptions: Raw CVE description strings.
            labels: Category name per description (must be in CATEGORIES).
            epochs: Training epochs.
            batch_size: Mini-batch size.
            lr: Adam learning rate.

        Returns:
            self
        """
        if len(descriptions) != len(labels):
            raise ValueError(
                f"descriptions and labels must have the same length, "
                f"got {len(descriptions)} vs {len(labels)}"
            )
        unknown = set(labels) - set(self._categories)
        if unknown:
            raise ValueError(f"Unknown label(s): {unknown}. Valid: {self._categories}")

        logger.info("Encoding %d descriptions with %s", len(descriptions), self._encoder)
        embeddings = self._encode_batch(descriptions)

        y = torch.tensor([self._label2idx[l] for l in labels], dtype=torch.long)
        X = torch.tensor(embeddings, dtype=torch.float32)

        optimizer = torch.optim.Adam(self._head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self._head.train()

        n = len(X)
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X[idx].to(self._device)
                yb = y[idx].to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._head(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            logger.debug("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, epoch_loss / n)

        self._head.eval()
        logger.info("Training complete")
        return self

    def predict(self, descriptions: list[str]) -> list[str]:
        """Return predicted category names for each description.

        Args:
            descriptions: Raw CVE description strings.

        Returns:
            List of category names (same length as input).
        """
        proba = self.predict_proba(descriptions)
        return [self._idx2label[int(i)] for i in np.argmax(proba, axis=1)]

    def predict_proba(self, descriptions: list[str]) -> np.ndarray:
        """Return probability distribution over categories.

        Args:
            descriptions: Raw CVE description strings.

        Returns:
            Array of shape (n_samples, n_categories) — rows sum to 1.
        """
        embeddings = self._encode_batch(descriptions)
        X = torch.tensor(embeddings, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._head(X)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_embeddings(self, descriptions: list[str]) -> np.ndarray:
        """Return raw sentence embeddings without classification.

        Args:
            descriptions: Raw CVE description strings.

        Returns:
            Array of shape (n_samples, embedding_dim).
        """
        return self._encode_batch(descriptions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_batch(self, descriptions: list[str]) -> np.ndarray:
        return self._encoder.encode(descriptions, show_progress_bar=False)
