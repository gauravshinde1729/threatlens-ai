"""Tests for src/models/text_classifier.py."""

import numpy as np
import pytest
import torch

from models.text_classifier import _EMBEDDING_DIM, CATEGORIES, CVETextClassifier

# ---------------------------------------------------------------------------
# Synthetic training data — 20 descriptions with known categories
# ---------------------------------------------------------------------------

_TRAIN_DESCRIPTIONS = [
    # injection (4)
    "SQL injection vulnerability allows attackers to execute arbitrary queries.",
    "Blind SQL injection in the login endpoint bypasses authentication checks.",
    "Command injection in the filename parameter enables OS command execution.",
    "LDAP injection in the search field exposes directory information.",
    # auth_bypass (3)
    "Authentication bypass due to missing session validation in admin panel.",
    "Improper authentication check allows unauthenticated access to dashboard.",
    "Auth bypass vulnerability via crafted JWT token grants admin privileges.",
    # memory_corruption (3)
    "Heap buffer overflow in the PNG parser leads to remote code execution.",
    "Use-after-free vulnerability in the network stack causes memory corruption.",
    "Stack-based buffer overflow when processing malformed input packets.",
    # info_disclosure (3)
    "Information disclosure exposes sensitive configuration data to remote users.",
    "Directory traversal allows reading arbitrary files outside the web root.",
    "Error messages reveal internal stack traces and database schema details.",
    # dos (3)
    "Denial of service via infinite loop when parsing crafted XML documents.",
    "Resource exhaustion vulnerability causes service crash under high load.",
    "Uncontrolled recursion leads to stack overflow and application crash.",
    # privilege_escalation (2)
    "Local privilege escalation allows non-root users to gain root access.",
    "Privilege escalation via SUID binary misconfiguration on Linux systems.",
    # other (2)
    "A misconfigured CORS policy allows cross-origin requests from any domain.",
    "Insecure default configuration exposes the management interface publicly.",
]

_TRAIN_LABELS = (
    ["injection"] * 4
    + ["auth_bypass"] * 3
    + ["memory_corruption"] * 3
    + ["info_disclosure"] * 3
    + ["dos"] * 3
    + ["privilege_escalation"] * 2
    + ["other"] * 2
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier() -> CVETextClassifier:
    """Shared fitted classifier — loading the model once per test module."""
    torch.manual_seed(42)
    np.random.seed(42)
    clf = CVETextClassifier()
    clf.fit(_TRAIN_DESCRIPTIONS, _TRAIN_LABELS, epochs=50)
    return clf


# ---------------------------------------------------------------------------
# test_fit_and_predict_returns_valid_categories
# ---------------------------------------------------------------------------


def test_fit_and_predict_returns_valid_categories(classifier):
    """predict() returns only valid category names for each input."""
    descriptions = [
        "SQL injection in the user input field.",
        "Buffer overflow when processing image files.",
        "Authentication bypass via malformed token.",
    ]
    predictions = classifier.predict(descriptions)

    assert len(predictions) == len(descriptions)
    for pred in predictions:
        assert pred in CATEGORIES, f"Unexpected category: {pred!r}"


def test_predict_single_description(classifier):
    """predict() works on a single-item list."""
    result = classifier.predict(["Remote code execution via crafted HTTP request."])
    assert len(result) == 1
    assert result[0] in CATEGORIES


def test_predict_obvious_injection(classifier):
    """A clear SQL injection description is predicted as 'injection'."""
    # Trained on injection examples — should confidently classify this
    result = classifier.predict(
        ["SQL injection allows executing arbitrary SQL commands in the database."]
    )
    assert result[0] == "injection"


def test_predict_on_training_data_is_consistent(classifier):
    """Predicting on training data returns consistent results (list, right length)."""
    preds = classifier.predict(_TRAIN_DESCRIPTIONS)
    assert len(preds) == len(_TRAIN_DESCRIPTIONS)
    assert all(p in CATEGORIES for p in preds)


# ---------------------------------------------------------------------------
# test_predict_proba_sums_to_one
# ---------------------------------------------------------------------------


def test_predict_proba_sums_to_one(classifier):
    """predict_proba() returns (n, n_categories) array; rows sum to 1."""
    descriptions = _TRAIN_DESCRIPTIONS[:5]
    proba = classifier.predict_proba(descriptions)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(descriptions), len(CATEGORIES))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(descriptions)), atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_argmax_matches_predict(classifier):
    """argmax of predict_proba matches the label returned by predict."""
    descriptions = _TRAIN_DESCRIPTIONS[:8]
    proba = classifier.predict_proba(descriptions)
    preds = classifier.predict(descriptions)

    for i, (p, pred) in enumerate(zip(proba, preds, strict=True)):
        expected_idx = CATEGORIES.index(pred)
        assert np.argmax(p) == expected_idx, (
            f"Row {i}: argmax={np.argmax(p)} ({CATEGORIES[np.argmax(p)]!r}) vs predict={pred!r}"
        )


def test_predict_proba_full_training_set(classifier):
    """predict_proba() handles the full training set without error."""
    proba = classifier.predict_proba(_TRAIN_DESCRIPTIONS)
    assert proba.shape == (len(_TRAIN_DESCRIPTIONS), len(CATEGORIES))


# ---------------------------------------------------------------------------
# test_get_embeddings_returns_correct_shape
# ---------------------------------------------------------------------------


def test_get_embeddings_returns_correct_shape(classifier):
    """get_embeddings() returns (n_samples, 384) float array."""
    descriptions = _TRAIN_DESCRIPTIONS[:6]
    embeddings = classifier.get_embeddings(descriptions)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(descriptions), _EMBEDDING_DIM)
    assert embeddings.dtype in (np.float32, np.float64)


def test_get_embeddings_single_item(classifier):
    """get_embeddings() works on a single description."""
    embeddings = classifier.get_embeddings(["A buffer overflow vulnerability."])
    assert embeddings.shape == (1, _EMBEDDING_DIM)


def test_get_embeddings_are_deterministic(classifier):
    """The same description always produces the same embedding."""
    desc = ["SQL injection allows arbitrary query execution."]
    emb1 = classifier.get_embeddings(desc)
    emb2 = classifier.get_embeddings(desc)
    np.testing.assert_allclose(emb1, emb2, atol=1e-5)


def test_get_embeddings_differ_across_descriptions(classifier):
    """Different descriptions produce different embeddings."""
    emb = classifier.get_embeddings(
        [
            "SQL injection vulnerability.",
            "Buffer overflow in memory parser.",
        ]
    )
    assert not np.allclose(emb[0], emb[1])


# ---------------------------------------------------------------------------
# fit() validation
# ---------------------------------------------------------------------------


def test_fit_raises_on_label_length_mismatch():
    """fit() raises ValueError when descriptions and labels have different lengths."""
    clf = CVETextClassifier()
    with pytest.raises(ValueError, match="same length"):
        clf.fit(["desc one", "desc two"], ["injection"])


def test_fit_raises_on_unknown_label():
    """fit() raises ValueError when an unrecognised label is provided."""
    clf = CVETextClassifier()
    with pytest.raises(ValueError, match="Unknown label"):
        clf.fit(["A SQL injection attack."], ["unknown_category"])
