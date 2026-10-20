"""
Pydantic schemas for the Handwritten Digit Feature and Similarity Analyzer API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class CanvasAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Data URL or base64 encoded image string")
    k_components: int = Field(16, ge=1, le=64, description="Number of principal components (1-64)")


class VectorAnalyzeRequest(BaseModel):
    matrix_8x8: Optional[List[List[float]]] = Field(None, description="8x8 matrix of pixel intensities [0, 16]")
    vector_64: Optional[List[float]] = Field(None, description="64-element flattened vector")
    k_components: int = Field(16, ge=1, le=64, description="Number of principal components")


class CandidateScore(BaseModel):
    digit: int
    similarity: float
    euclidean_distance: float
    rank: int


class MathPipelineResponse(BaseModel):
    predicted_digit: int
    predicted_similarity: float
    k_components: int
    cumulative_variance_explained: float
    reconstruction_error: float
    idempotency_error: float
    norm_x: float
    matrix_8x8: List[List[float]]
    vector_x: List[float]
    normalized_vector_v: List[float]
    mean_vector_mu: List[float]
    centered_vector_vc: List[float]
    projected_z: List[float]
    projected_z_2d: List[float]
    reconstructed_matrix_8x8: List[List[float]]
    similarities: Dict[str, float]
    euclidean_distances: Dict[str, float]
    top_candidates: List[CandidateScore]
    is_blank: bool = False
    message: Optional[str] = None


class ClassMetric(BaseModel):
    digit: int
    precision: float
    recall: float
    f1_score: float
    support: int


class EvaluationResponse(BaseModel):
    k_components: int
    accuracy: float
    total_test_samples: int
    correct_predictions: int
    confusion_matrix: List[List[int]]
    per_class_metrics: List[ClassMetric]
    cumulative_variance: float


class PrototypeItem(BaseModel):
    digit: int
    projected_vector: List[float]
    reconstructed_matrix_8x8: List[List[float]]
    norm: float


class ModelInfoResponse(BaseModel):
    dataset: str
    total_samples: int
    train_samples: int
    test_samples: int
    input_dimensions: List[int]
    feature_vector_size: int
    num_classes: int
    eigenvalues: List[float]
    explained_variance_ratio: List[float]
    cumulative_variance_ratio: List[float]
