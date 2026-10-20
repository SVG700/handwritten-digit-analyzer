"""
FastAPI application for Handwritten Digit Feature and Similarity Analyzer (MAT2303).
"""

from typing import Dict, Any, List
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.math_pipeline import DigitFeatureAnalyzer
from backend.app.preprocessor import base64_to_image, bytes_to_image, preprocess_image_to_8x8
from backend.app.schemas import (
    CanvasAnalyzeRequest,
    VectorAnalyzeRequest,
    MathPipelineResponse,
    EvaluationResponse,
    ModelInfoResponse,
    PrototypeItem,
)

# Global singleton instance of the mathematical pipeline
analyzer: DigitFeatureAnalyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and fit model on startup once
    global analyzer
    if analyzer is None:
        analyzer = DigitFeatureAnalyzer(random_state=42, test_size=0.2)
    yield


app = FastAPI(
    title="Handwritten Digit Feature and Similarity Analyzer API",
    description="Linear Algebra & Vector Calculus (MAT2303) CA3 Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_analyzer() -> DigitFeatureAnalyzer:
    global analyzer
    if analyzer is None:
        analyzer = DigitFeatureAnalyzer(random_state=42, test_size=0.2)
    return analyzer


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint confirming backend and model readiness."""
    eng = get_analyzer()
    return {
        "status": "healthy",
        "service": "Handwritten Digit Feature and Similarity Analyzer",
        "model_fitted": str(eng.is_fitted),
        "dataset": "scikit-learn 8x8 digits",
    }


@app.get("/api/model/info", response_model=ModelInfoResponse)
def get_model_information() -> Dict[str, Any]:
    """Returns dataset metadata, dimensionality, and PCA explained variance distribution."""
    eng = get_analyzer()
    return eng.get_model_info()


@app.post("/api/analyze/canvas", response_model=MathPipelineResponse)
def analyze_canvas(request: CanvasAnalyzeRequest) -> Dict[str, Any]:
    """
    Processes a digit drawn on the HTML5 canvas.
    Takes base64 image data, preprocesses it to 8x8 grayscale,
    and runs the 6-stage linear algebra pipeline.
    """
    eng = get_analyzer()
    try:
        pil_img = base64_to_image(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image encoding: {str(e)}")

    matrix_8x8, is_blank = preprocess_image_to_8x8(pil_img)
    if is_blank:
        analysis = eng.analyze(np.zeros((8, 8)), k=request.k_components)
        analysis["is_blank"] = True
        analysis["message"] = "Canvas appears blank. Please draw a digit before analyzing."
        return analysis

    analysis = eng.analyze(matrix_8x8, k=request.k_components)
    analysis["is_blank"] = False
    return analysis


@app.post("/api/analyze/upload", response_model=MathPipelineResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    k_components: int = Query(16, ge=1, le=64)
) -> Dict[str, Any]:
    """
    Processes an uploaded image file (PNG, JPG, BMP, etc.).
    Preprocesses to 8x8 grayscale and executes pipeline.
    """
    eng = get_analyzer()
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (PNG/JPG/WEBP/BMP).")

    contents = await file.read()
    try:
        pil_img = bytes_to_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    matrix_8x8, is_blank = preprocess_image_to_8x8(pil_img)
    if is_blank:
        analysis = eng.analyze(np.zeros((8, 8)), k=k_components)
        analysis["is_blank"] = True
        analysis["message"] = "Uploaded image appears completely blank or unreadable."
        return analysis

    analysis = eng.analyze(matrix_8x8, k=k_components)
    analysis["is_blank"] = False
    return analysis


@app.post("/api/analyze/vector", response_model=MathPipelineResponse)
def analyze_vector(request: VectorAnalyzeRequest) -> Dict[str, Any]:
    """
    Processes a direct 8x8 matrix or 64-element numerical vector.
    Ideal for testing, reproducible benchmarks, and viva demonstrations.
    """
    eng = get_analyzer()
    if request.matrix_8x8 is not None:
        mat = np.array(request.matrix_8x8, dtype=np.float64)
        if mat.shape != (8, 8):
            raise HTTPException(status_code=400, detail=f"Matrix must be 8x8, got {mat.shape}")
    elif request.vector_64 is not None:
        vec = np.array(request.vector_64, dtype=np.float64)
        if vec.shape != (64,):
            raise HTTPException(status_code=400, detail=f"Vector must have 64 elements, got {vec.shape}")
        mat = vec.reshape((8, 8))
    else:
        raise HTTPException(status_code=400, detail="Provide either matrix_8x8 or vector_64.")

    analysis = eng.analyze(mat, k=request.k_components)
    analysis["is_blank"] = bool(np.all(mat == 0))
    return analysis


@app.get("/api/evaluate", response_model=EvaluationResponse)
def evaluate_model(k_components: int = Query(16, ge=1, le=64)) -> Dict[str, Any]:
    """
    Evaluates classification performance on the held-out test split
    for the specified number of principal components k.
    """
    eng = get_analyzer()
    return eng.evaluate(k=k_components)


@app.get("/api/sample/{digit}")
def get_sample(digit: int) -> Dict[str, Any]:
    """
    Fetches a real, held-out test set sample for digit 0-9.
    Enables quick one-click testing in the web UI.
    """
    if digit < 0 or digit > 9:
        raise HTTPException(status_code=400, detail="Digit must be between 0 and 9.")
    eng = get_analyzer()
    return eng.get_sample_test_digit(digit)


@app.get("/api/prototypes", response_model=List[PrototypeItem])
def get_prototypes(k_components: int = Query(16, ge=1, le=64)) -> List[Dict[str, Any]]:
    """Returns the 10 class prototype vectors in reduced space and reconstructed 8x8 matrices."""
    eng = get_analyzer()
    return eng.get_prototypes_info(k=k_components)
