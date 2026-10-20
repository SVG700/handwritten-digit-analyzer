"""
Unit tests for the mathematical pipeline:
- Matrix conversion & shape verification
- Vectorization and Euclidean norm
- Zero-vector handling
- Orthogonality of eigenvectors (W_k^T W_k = I)
- Projection idempotency (P^2 = P) and symmetry (P^T = P)
- Cosine similarity boundaries [-1, 1]
- Prediction argmax logic
"""

import numpy as np
import pytest
from backend.app.math_pipeline import (
    vectorize_matrix,
    euclidean_norm,
    normalize_vector,
    cosine_similarity,
    DigitFeatureAnalyzer,
)


def test_vectorize_matrix_valid():
    mat = np.arange(64, dtype=float).reshape((8, 8))
    vec = vectorize_matrix(mat)
    assert vec.shape == (64,)
    assert vec[0] == 0.0
    assert vec[63] == 63.0


def test_vectorize_matrix_invalid_shape():
    with pytest.raises(ValueError):
        vectorize_matrix(np.zeros((7, 8)))


def test_euclidean_norm_calculation():
    vec = np.array([3.0, 4.0, 0.0])
    norm = euclidean_norm(vec)
    assert pytest.approx(norm, 1e-6) == 5.0


def test_zero_vector_normalization():
    zero_vec = np.zeros(64)
    v, norm = normalize_vector(zero_vec)
    assert norm == 0.0
    assert np.all(v == 0.0)


def test_unit_vector_normalization():
    vec = np.random.RandomState(42).randn(64)
    v, norm = normalize_vector(vec)
    assert norm > 0
    assert pytest.approx(euclidean_norm(v), 1e-6) == 1.0


def test_cosine_similarity_properties():
    rng = np.random.RandomState(123)
    a = rng.randn(16)
    b = rng.randn(16)
    
    # Self-similarity is 1
    assert pytest.approx(cosine_similarity(a, a), 1e-6) == 1.0
    # Opposite vector similarity is -1
    assert pytest.approx(cosine_similarity(a, -a), 1e-6) == -1.0
    # Orthogonal vectors have similarity 0
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(e1, e2), 1e-6) == 0.0
    # Zero vector returns 0
    assert cosine_similarity(a, np.zeros(16)) == 0.0


def test_pca_orthogonality():
    analyzer = DigitFeatureAnalyzer(random_state=42)
    k = 16
    W_k = analyzer.get_projection_matrix(k)
    assert W_k.shape == (64, k)
    
    # W_k^T * W_k should be Identity matrix I_k
    gram = np.dot(W_k.T, W_k)
    identity = np.eye(k)
    max_err = np.max(np.abs(gram - identity))
    assert max_err < 1e-6, f"Orthogonality violation: {max_err}"


def test_projection_matrix_idempotency_and_symmetry():
    analyzer = DigitFeatureAnalyzer(random_state=42)
    k = 12
    W_k = analyzer.get_projection_matrix(k)
    P_k = np.dot(W_k, W_k.T)  # Shape (64, 64)
    
    # 1. Symmetry: P^T = P
    sym_diff = np.max(np.abs(P_k - P_k.T))
    assert sym_diff < 1e-6, f"Projection matrix is not symmetric: {sym_diff}"
    
    # 2. Idempotency: P^2 = P
    P_sq = np.dot(P_k, P_k)
    idem_diff = np.max(np.abs(P_sq - P_k))
    assert idem_diff < 1e-6, f"Projection matrix is not idempotent: {idem_diff}"


def test_pipeline_analysis_output_structure():
    analyzer = DigitFeatureAnalyzer(random_state=42)
    sample = analyzer.get_sample_test_digit(3)
    mat = np.array(sample["matrix_8x8"])
    
    res = analyzer.analyze(mat, k=16)
    assert res["predicted_digit"] in range(10)
    assert len(res["similarities"]) == 10
    assert len(res["top_candidates"]) == 10
    assert res["k_components"] == 16
    assert res["reconstruction_error"] >= 0.0
    assert res["idempotency_error"] < 1e-6


def test_pipeline_test_evaluation_accuracy():
    analyzer = DigitFeatureAnalyzer(random_state=42)
    eval_res = analyzer.evaluate(k=24)
    # Cosine similarity nearest prototype in 24D PCA space should achieve high accuracy (>85%)
    assert eval_res["accuracy"] > 0.85
    assert len(eval_res["confusion_matrix"]) == 10
    assert len(eval_res["per_class_metrics"]) == 10
