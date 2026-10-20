"""
Integration tests for FastAPI endpoints:
- Health check
- Model info
- Canvas analysis (valid and blank)
- Direct vector / matrix analysis
- Model evaluation
- Digit sample retrieval
"""

import base64
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_fitted"] == "True"


def test_model_info_endpoint():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_vector_size"] == 64
    assert data["num_classes"] == 10
    assert len(data["eigenvalues"]) == 64
    assert len(data["explained_variance_ratio"]) == 64


def test_sample_digit_endpoint():
    for d in [0, 5, 8]:
        response = client.get(f"/api/sample/{d}")
        assert response.status_code == 200
        data = response.json()
        assert data["digit"] == d
        assert len(data["matrix_8x8"]) == 8
        assert len(data["matrix_8x8"][0]) == 8


def test_analyze_vector_endpoint():
    # Use a known sample of digit 7
    sample_resp = client.get("/api/sample/7")
    mat = sample_resp.json()["matrix_8x8"]
    
    payload = {
        "matrix_8x8": mat,
        "k_components": 16
    }
    response = client.post("/api/analyze/vector", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["k_components"] == 16
    assert "predicted_digit" in data
    assert len(data["similarities"]) == 10
    assert data["reconstruction_error"] >= 0.0


def test_analyze_canvas_endpoint():
    # Create a small PIL image with a white shape on black background
    img = Image.new("L", (100, 100), color=0)
    # Draw simple vertical bar
    for y in range(20, 80):
        for x in range(45, 55):
            img.putpixel((x, y), 255)
            
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    
    payload = {
        "image_base64": b64,
        "k_components": 20
    }
    response = client.post("/api/analyze/canvas", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert not data["is_blank"]
    assert data["predicted_digit"] in range(10)


def test_analyze_blank_canvas_endpoint():
    # Blank black image
    img = Image.new("L", (50, 50), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    
    payload = {
        "image_base64": b64,
        "k_components": 16
    }
    response = client.post("/api/analyze/canvas", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_blank"] is True


def test_evaluate_endpoint():
    response = client.get("/api/evaluate?k_components=16")
    assert response.status_code == 200
    data = response.json()
    assert data["k_components"] == 16
    assert 0.0 <= data["accuracy"] <= 1.0
    assert len(data["confusion_matrix"]) == 10
    assert len(data["per_class_metrics"]) == 10


def test_prototypes_endpoint():
    response = client.get("/api/prototypes?k_components=16")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    assert data[0]["digit"] == 0
    assert len(data[0]["projected_vector"]) == 16
