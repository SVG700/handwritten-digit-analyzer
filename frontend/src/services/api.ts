import type {
  MathPipelineResponse,
  EvaluationResponse,
  ModelInfoResponse,
  PrototypeItem,
} from '../types';

const API_BASE = '/api';

export async function checkHealth(): Promise<{ status: string; model_fitted: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error(`Failed to load model info: ${res.statusText}`);
  return res.json();
}

export async function analyzeCanvas(
  imageBase64: string,
  kComponents: number
): Promise<MathPipelineResponse> {
  const res = await fetch(`${API_BASE}/analyze/canvas`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      k_components: kComponents,
    }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed with status ${res.status}`);
  }
  return res.json();
}

export async function analyzeUpload(
  file: File,
  kComponents: number
): Promise<MathPipelineResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/analyze/upload?k_components=${kComponents}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload analysis failed: ${res.status}`);
  }
  return res.json();
}

export async function analyzeVector(
  matrix8x8: number[][],
  kComponents: number
): Promise<MathPipelineResponse> {
  const res = await fetch(`${API_BASE}/analyze/vector`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      matrix_8x8: matrix8x8,
      k_components: kComponents,
    }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Vector analysis failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchEvaluation(kComponents: number = 16): Promise<EvaluationResponse> {
  const res = await fetch(`${API_BASE}/evaluate?k_components=${kComponents}`);
  if (!res.ok) throw new Error(`Failed to load evaluation metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchSampleDigit(
  digit: number
): Promise<{ digit: number; matrix_8x8: number[][]; vector: number[] }> {
  const res = await fetch(`${API_BASE}/sample/${digit}`);
  if (!res.ok) throw new Error(`Failed to load test sample for digit ${digit}`);
  return res.json();
}

export async function fetchPrototypes(kComponents: number = 16): Promise<PrototypeItem[]> {
  const res = await fetch(`${API_BASE}/prototypes?k_components=${kComponents}`);
  if (!res.ok) throw new Error(`Failed to load prototypes: ${res.statusText}`);
  return res.json();
}
