"""
Linear Algebra & Vector Calculus (MAT2303)
Mathematical Pipeline: Handwritten Digit Feature and Similarity Analyzer

This module implements the complete 6-stage mathematical pipeline:
1. Matrix representation: M in R^(8x8)
2. Vectorization & Euclidean normalization: x = vec(M) in R^64, v = x / ||x||_2
3. PCA / SVD: Covariance C = X_c^T X_c / (N-1), Spectral decomposition C = V Lambda V^T
4. Orthogonal projection: W_k in R^(64xk), z = W_k^T (v - mu), P_k = W_k W_k^T
5. Inner-product similarity: Cosine similarity sim(z, p_c) = <z, p_c> / (||z|| * ||p_c||)
6. Prediction: y_hat = argmax_c sim(z, p_c)

All fitting is performed strictly on training data without data leakage.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def vectorize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Stage 1 -> 2: Flatten an 8x8 matrix into a 64-dimensional column/row vector.
    
    Args:
        matrix: 2D NumPy array of shape (8, 8).
    Returns:
        1D NumPy array of shape (64,), flattened row-major.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (8, 8):
        raise ValueError(f"Expected 8x8 matrix, got shape {mat.shape}")
    return mat.flatten()


def euclidean_norm(vector: np.ndarray) -> float:
    """
    Computes the L2 (Euclidean) norm: ||x||_2 = sqrt(sum(x_i^2)).
    """
    vec = np.asarray(vector, dtype=np.float64).flatten()
    return float(np.sqrt(np.sum(vec ** 2)))


def normalize_vector(vector: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Stage 2: Euclidean normalization with explicit zero-vector handling.
    
    v = x / ||x||_2 if ||x||_2 > 0 else 0
    
    Args:
        vector: 1D array of shape (64,).
    Returns:
        Tuple of (normalized_vector, original_norm).
    """
    vec = np.asarray(vector, dtype=np.float64).flatten()
    norm = euclidean_norm(vec)
    if norm < 1e-12:
        return np.zeros_like(vec), 0.0
    return vec / norm, norm


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Stage 5: Normalized inner product (Cosine similarity).
    
    sim(v1, v2) = <v1, v2> / (||v1||_2 * ||v2||_2) in [-1, 1]
    Handles zero vectors safely by returning 0.0.
    """
    a = np.asarray(v1, dtype=np.float64).flatten()
    b = np.asarray(v2, dtype=np.float64).flatten()
    norm_a = euclidean_norm(a)
    norm_b = euclidean_norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    dot = float(np.dot(a, b))
    # Clip numerical precision fluctuations beyond [-1.0, 1.0]
    return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))


class DigitFeatureAnalyzer:
    """
    Core mathematical engine for handwritten digit analysis.
    Trained strictly on training split of scikit-learn digits dataset (8x8 grayscale).
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size = test_size
        
        # Raw splits
        self.X_train_raw: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test_raw: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        # Normalized training features
        self.X_train_norm: Optional[np.ndarray] = None
        self.X_test_norm: Optional[np.ndarray] = None
        
        # PCA / SVD parameters
        self.mean_vector_: Optional[np.ndarray] = None        # mu in R^64
        self.cov_matrix_: Optional[np.ndarray] = None         # C in R^(64x64)
        self.eigenvalues_: Optional[np.ndarray] = None        # lambda_1 >= ... >= lambda_64
        self.eigenvectors_: Optional[np.ndarray] = None       # V in R^(64x64), columns are u_i
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None
        
        # Class prototype cache
        self.is_fitted: bool = False
        
        # Load and fit automatically on instantiation
        self.fit_dataset()

    def fit_dataset(self) -> None:
        """Loads scikit-learn 8x8 digits dataset and fits the mathematical pipeline."""
        digits = load_digits()
        X = digits.data.astype(np.float64)  # Shape (1797, 64)
        y = digits.target.astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self.fit(X_train, y_train, X_test, y_test)

    def fit(
        self,
        X_train_raw: np.ndarray,
        y_train: np.ndarray,
        X_test_raw: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> "DigitFeatureAnalyzer":
        """
        Fits vectorization normalization, PCA/SVD decomposition, and prototypes
        using training data ONLY.
        """
        self.X_train_raw = np.asarray(X_train_raw, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)

        N_train, D = self.X_train_raw.shape
        if D != 64:
            raise ValueError(f"Features must have dimension 64, got {D}")

        # 1. Normalize each training vector
        self.X_train_norm = np.zeros_like(self.X_train_raw)
        for i in range(N_train):
            norm_v, _ = normalize_vector(self.X_train_raw[i])
            self.X_train_norm[i] = norm_v

        # 2. Compute training mean vector: mu = 1/N * sum(v_i)
        self.mean_vector_ = np.mean(self.X_train_norm, axis=0)

        # 3. Centered training matrix: X_c = X_train_norm - mu
        X_c = self.X_train_norm - self.mean_vector_

        # 4. Sample Covariance matrix: C = 1/(N - 1) * X_c^T * X_c
        self.cov_matrix_ = np.dot(X_c.T, X_c) / (N_train - 1)

        # 5. Eigendecomposition: C * u_i = lambda_i * u_i
        # Using eigh since covariance matrix is symmetric positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(self.cov_matrix_)

        # Sort in descending order of eigenvalues
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[idx]
        self.eigenvectors_ = eigvecs[:, idx]  # Columns are eigenvectors u_j

        total_var = float(np.sum(self.eigenvalues_))
        if total_var > 0:
            self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.eigenvalues_)
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Also prepare test set if provided
        if X_test_raw is not None and y_test is not None:
            self.X_test_raw = np.asarray(X_test_raw, dtype=np.float64)
            self.y_test = np.asarray(y_test, dtype=np.int64)
            self.X_test_norm = np.zeros_like(self.X_test_raw)
            for i in range(len(self.X_test_raw)):
                norm_v, _ = normalize_vector(self.X_test_raw[i])
                self.X_test_norm[i] = norm_v

        self.is_fitted = True
        return self

    def get_projection_matrix(self, k: int) -> np.ndarray:
        """
        Returns basis matrix W_k in R^(64 x k), where columns are top-k eigenvectors.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        k = max(1, min(64, int(k)))
        return self.eigenvectors_[:, :k]

    def compute_class_prototypes(self, k: int = 16) -> Dict[int, np.ndarray]:
        """
        Computes digit class prototypes in the k-dimensional projected feature space.
        Prototype for class c is the mean projected vector of training samples with label c.
        
        p_c = 1/|S_c| * sum_{i in S_c} z_i in R^k
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        W_k = self.get_projection_matrix(k)
        
        # Center training data
        X_c = self.X_train_norm - self.mean_vector_
        # Project into R^k: Z_train = X_c * W_k
        Z_train = np.dot(X_c, W_k)  # (N_train, k)

        prototypes = {}
        for c in range(10):
            mask = (self.y_train == c)
            if np.any(mask):
                prototypes[c] = np.mean(Z_train[mask], axis=0)
            else:
                prototypes[c] = np.zeros(k)
        return prototypes

    def analyze(self, matrix_8x8: np.ndarray, k: int = 16) -> Dict[str, Any]:
        """
        Executes the complete 6-stage mathematical pipeline on an 8x8 input matrix.
        
        Stage 1: Grayscale pixel matrix M in R^(8x8)
        Stage 2: Vectorization x = vec(M) and Euclidean normalization v = x / ||x||_2
        Stage 3: Feature subspace reduction using top-k components W_k
        Stage 4: Orthogonal projection z = W_k^T (v - mu) and reconstruction v_hat
        Stage 5: Cosine similarity with 10 digit class prototypes
        Stage 6: Prediction y_hat = argmax_c sim(z, p_c)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        k = max(1, min(64, int(k)))
        W_k = self.get_projection_matrix(k)

        # Stage 1: Matrix Representation
        M = np.asarray(matrix_8x8, dtype=np.float64)
        if M.shape != (8, 8):
            raise ValueError(f"Input matrix must have shape (8, 8), got {M.shape}")

        # Stage 2: Vectorization & Normalization
        x = vectorize_matrix(M)
        norm_x = euclidean_norm(x)
        v, _ = normalize_vector(x)

        # Stage 3 & 4: Centering & Orthogonal Projection
        # v_c = v - mu
        v_c = v - self.mean_vector_
        
        # z = W_k^T * v_c in R^k
        z = np.dot(W_k.T, v_c)

        # Orthogonal projection back to centered space: v_c_hat = W_k * z = W_k * W_k^T * v_c
        v_c_hat = np.dot(W_k, z)
        # Reconstructed normalized vector: v_hat = v_c_hat + mu
        v_hat = v_c_hat + self.mean_vector_
        # Reconstructed raw approximation
        x_hat = v_hat * (norm_x if norm_x > 0 else 1.0)
        M_hat = x_hat.reshape((8, 8))

        # Projection error: ||v_c - v_c_hat||_2
        reconstruction_error = float(euclidean_norm(v_c - v_c_hat))
        
        # Projection operator P_k = W_k * W_k^T
        # Verify idempotency P_k^2 = P_k for math demonstration
        P_k = np.dot(W_k, W_k.T)
        P_k_squared = np.dot(P_k, P_k)
        idempotency_error = float(np.max(np.abs(P_k_squared - P_k)))

        # Stage 5: Inner-product / Cosine Similarity with prototypes
        prototypes = self.compute_class_prototypes(k)
        similarities = {}
        euclidean_distances = {}

        norm_z = euclidean_norm(z)

        for c in range(10):
            proto_c = prototypes[c]
            sim = cosine_similarity(z, proto_c)
            dist = float(euclidean_norm(z - proto_c))
            similarities[c] = sim
            euclidean_distances[c] = dist

        # Stage 6: Prediction
        predicted_digit = int(max(similarities.keys(), key=lambda c: similarities[c]))
        predicted_similarity = similarities[predicted_digit]

        # Top 3 candidates
        sorted_digits = sorted(similarities.keys(), key=lambda c: similarities[c], reverse=True)
        top_candidates = [
            {
                "digit": c,
                "similarity": round(similarities[c], 5),
                "euclidean_distance": round(euclidean_distances[c], 5),
                "rank": rank + 1,
            }
            for rank, c in enumerate(sorted_digits)
        ]

        # 2D projection for easy coordinate plotting (first 2 components)
        W_2 = self.get_projection_matrix(2)
        z_2d = np.dot(W_2.T, v_c).tolist()

        return {
            "matrix_8x8": M.tolist(),
            "vector_x": x.tolist(),
            "norm_x": round(norm_x, 6),
            "normalized_vector_v": v.tolist(),
            "mean_vector_mu": self.mean_vector_.tolist(),
            "centered_vector_vc": v_c.tolist(),
            "projected_z": z.tolist(),
            "projected_z_2d": z_2d,
            "reconstructed_v": v_hat.tolist(),
            "reconstructed_matrix_8x8": M_hat.tolist(),
            "reconstruction_error": round(reconstruction_error, 6),
            "idempotency_error": round(idempotency_error, 10),
            "k_components": k,
            "cumulative_variance_explained": round(float(self.cumulative_variance_ratio_[k - 1]), 6),
            "similarities": {str(c): round(s, 5) for c, s in similarities.items()},
            "euclidean_distances": {str(c): round(d, 5) for c, d in euclidean_distances.items()},
            "predicted_digit": predicted_digit,
            "predicted_similarity": round(predicted_similarity, 5),
            "top_candidates": top_candidates,
        }

    def evaluate(self, k: int = 16) -> Dict[str, Any]:
        """
        Evaluates the classifier on the held-out test split.
        Returns accuracy, confusion matrix, precision, recall, and F1-score per class.
        No data leakage: test samples are projected using W_k and mu fitted only on train.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        k = max(1, min(64, int(k)))
        W_k = self.get_projection_matrix(k)
        prototypes = self.compute_class_prototypes(k)

        # Center test data with training mean: X_test_c = X_test_norm - mu
        X_test_c = self.X_test_norm - self.mean_vector_
        # Project test data: Z_test = X_test_c * W_k
        Z_test = np.dot(X_test_c, W_k)

        # Compute cosine similarity for each test sample against each prototype
        N_test = len(self.y_test)
        y_pred = np.zeros(N_test, dtype=np.int64)
        
        # Precompute prototype norms
        proto_matrix = np.array([prototypes[c] for c in range(10)])  # (10, k)
        proto_norms = np.linalg.norm(proto_matrix, axis=1)  # (10,)
        proto_norms[proto_norms < 1e-12] = 1.0
        normalized_prototypes = proto_matrix / proto_norms[:, np.newaxis]  # (10, k)

        for i in range(N_test):
            z_i = Z_test[i]
            norm_z = np.linalg.norm(z_i)
            if norm_z < 1e-12:
                sims = np.zeros(10)
            else:
                z_unit = z_i / norm_z
                sims = np.dot(normalized_prototypes, z_unit)
            y_pred[i] = int(np.argmax(sims))

        # Overall accuracy
        correct = np.sum(y_pred == self.y_test)
        accuracy = float(correct / N_test)

        # Confusion matrix (10x10)
        conf_matrix = np.zeros((10, 10), dtype=int)
        for true_label, pred_label in zip(self.y_test, y_pred):
            conf_matrix[true_label, pred_label] += 1

        # Per-class metrics
        per_class_metrics = []
        for c in range(10):
            tp = int(conf_matrix[c, c])
            fp = int(np.sum(conf_matrix[:, c]) - tp)
            fn = int(np.sum(conf_matrix[c, :]) - tp)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            support = int(np.sum(self.y_test == c))

            per_class_metrics.append({
                "digit": c,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": support,
            })

        return {
            "k_components": k,
            "accuracy": round(accuracy, 4),
            "total_test_samples": N_test,
            "correct_predictions": int(correct),
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_metrics": per_class_metrics,
            "cumulative_variance": round(float(self.cumulative_variance_ratio_[k - 1]), 4),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Provides metadata about the dataset, dimensions, and spectral variance."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
            
        return {
            "dataset": "scikit-learn load_digits (8x8 grayscale)",
            "total_samples": len(self.X_train_raw) + len(self.X_test_raw),
            "train_samples": len(self.X_train_raw),
            "test_samples": len(self.X_test_raw),
            "input_dimensions": [8, 8],
            "feature_vector_size": 64,
            "num_classes": 10,
            "eigenvalues": [round(float(e), 6) for e in self.eigenvalues_],
            "explained_variance_ratio": [round(float(v), 6) for v in self.explained_variance_ratio_],
            "cumulative_variance_ratio": [round(float(c), 6) for c in self.cumulative_variance_ratio_],
        }

    def get_sample_test_digit(self, digit: int) -> Dict[str, Any]:
        """Returns a real test set sample for the requested digit."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        digit = int(digit) % 10
        indices = np.where(self.y_test == digit)[0]
        if len(indices) == 0:
            raise ValueError(f"No sample found for digit {digit}")
        idx = indices[0]
        raw_vec = self.X_test_raw[idx]
        mat = raw_vec.reshape((8, 8))
        return {
            "digit": digit,
            "matrix_8x8": mat.tolist(),
            "vector": raw_vec.tolist(),
        }

    def get_prototypes_info(self, k: int = 16) -> List[Dict[str, Any]]:
        """Returns the 10 class prototype representations."""
        prototypes = self.compute_class_prototypes(k)
        W_k = self.get_projection_matrix(k)
        
        results = []
        for c in range(10):
            proto_z = prototypes[c]
            # Reconstruct into 8x8 normalized space: proto_v = W_k * proto_z + mu
            proto_v = np.dot(W_k, proto_z) + self.mean_vector_
            proto_mat = np.clip(proto_v.reshape((8, 8)) * 16.0, 0, 16.0)
            results.append({
                "digit": c,
                "projected_vector": [round(float(val), 4) for val in proto_z],
                "reconstructed_matrix_8x8": proto_mat.tolist(),
                "norm": round(euclidean_norm(proto_z), 4),
            })
        return results
