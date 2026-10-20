export interface CandidateScore {
  digit: number;
  similarity: number;
  euclidean_distance: number;
  rank: number;
}

export interface MathPipelineResponse {
  predicted_digit: number;
  predicted_similarity: number;
  k_components: number;
  cumulative_variance_explained: number;
  reconstruction_error: number;
  idempotency_error: number;
  norm_x: number;
  matrix_8x8: number[][];
  vector_x: number[];
  normalized_vector_v: number[];
  mean_vector_mu: number[];
  centered_vector_vc: number[];
  projected_z: number[];
  projected_z_2d: number[];
  reconstructed_matrix_8x8: number[][];
  similarities: Record<string, number>;
  euclidean_distances: Record<string, number>;
  top_candidates: CandidateScore[];
  is_blank?: boolean;
  message?: string;
}

export interface ClassMetric {
  digit: number;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface EvaluationResponse {
  k_components: number;
  accuracy: number;
  total_test_samples: number;
  correct_predictions: number;
  confusion_matrix: number[][];
  per_class_metrics: ClassMetric[];
  cumulative_variance: number;
}

export interface ModelInfoResponse {
  dataset: string;
  total_samples: number;
  train_samples: number;
  test_samples: number;
  input_dimensions: number[];
  feature_vector_size: number;
  num_classes: number;
  eigenvalues: number[];
  explained_variance_ratio: number[];
  cumulative_variance_ratio: number[];
}

export interface PrototypeItem {
  digit: number;
  projected_vector: number[];
  reconstructed_matrix_8x8: number[][];
  norm: number;
}
