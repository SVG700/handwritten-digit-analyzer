import React, { useState, useEffect } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import { CanvasDrawer } from '../components/CanvasDrawer';
import { SimilarityChart } from '../components/SimilarityChart';
import { ReconstructionView } from '../components/ReconstructionView';
import { ProjectionPlot } from '../components/ProjectionPlot';
import type { MathPipelineResponse } from '../types';
import { analyzeCanvas, analyzeUpload, analyzeVector, fetchSampleDigit } from '../services/api';

interface AnalyzerPageProps {
  setActiveTab: (tab: string) => void;
  latestResult: MathPipelineResponse | null;
  setLatestResult: (result: MathPipelineResponse | null) => void;
}

export const AnalyzerPage: React.FC<AnalyzerPageProps> = ({
  setActiveTab,
  latestResult,
  setLatestResult,
}) => {
  const [kComponents, setKComponents] = useState(16);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load an initial sample (e.g. digit 3) on first load if no result exists yet
  useEffect(() => {
    if (!latestResult) {
      handleLoadInitialSample();
    }
  }, []);

  const handleLoadInitialSample = async () => {
    try {
      setIsLoading(true);
      const sample = await fetchSampleDigit(3);
      const res = await analyzeVector(sample.matrix_8x8, kComponents);
      setLatestResult(res);
    } catch (err: any) {
      console.warn('Backend not ready yet for initial sample, waiting for user action:', err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeCanvas = async (base64: string, k: number) => {
    try {
      setIsLoading(true);
      setErrorMessage(null);
      const res = await analyzeCanvas(base64, k);
      setLatestResult(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to analyze canvas drawing.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeUpload = async (file: File, k: number) => {
    try {
      setIsLoading(true);
      setErrorMessage(null);
      const res = await analyzeUpload(file, k);
      setLatestResult(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to analyze uploaded image.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeMatrix = async (matrix: number[][], k: number) => {
    try {
      setIsLoading(true);
      setErrorMessage(null);
      const res = await analyzeVector(matrix, k);
      setLatestResult(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to analyze matrix.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-4">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
          <span>Digit Feature & Similarity Analyzer</span>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-cyan-950 border border-cyan-800 text-cyan-400">
            Live Math Pipeline
          </span>
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Draw or upload any digit to inspect its vector transformation, PCA orthogonal projection, and cosine similarities.
        </p>
      </div>

      {/* Error alert */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-red-950/80 border border-red-800 text-red-200 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Input component */}
      <CanvasDrawer
        onAnalyzeCanvas={handleAnalyzeCanvas}
        onAnalyzeUpload={handleAnalyzeUpload}
        onAnalyzeMatrix={handleAnalyzeMatrix}
        isLoading={isLoading}
        kComponents={kComponents}
        setKComponents={setKComponents}
      />

      {/* Results View */}
      {latestResult && (
        <div className="space-y-8 animate-fadeIn">
          {/* Winner Banner */}
          <div className="bg-gradient-to-r from-violet-950/80 via-[#0f172a] to-cyan-950/80 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl relative overflow-hidden">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-6">
                {/* Big Predicted Digit Display */}
                <div className="w-24 h-24 sm:w-28 sm:h-28 rounded-2xl bg-gradient-to-br from-violet-600 to-cyan-500 p-0.5 shadow-xl shadow-violet-600/30">
                  <div className="w-full h-full bg-[#070a13] rounded-[14px] flex flex-col items-center justify-center">
                    <span className="text-xs uppercase font-mono text-cyan-400 font-bold">Predicted</span>
                    <span className="text-5xl sm:text-6xl font-black text-white tracking-tight">
                      {latestResult.predicted_digit}
                    </span>
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      Classified via Cosine Argmax
                    </span>
                    <span className="text-xs text-slate-400 font-mono">k = {latestResult.k_components}</span>
                  </div>
                  <h3 className="text-2xl font-bold text-white">
                    Digit {latestResult.predicted_digit} Detected
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 max-w-md">
                    Maximum normalized inner product: sim(z, p_{latestResult.predicted_digit}) ={' '}
                    <strong className="text-cyan-400 font-mono">
                      {latestResult.predicted_similarity.toFixed(4)}
                    </strong>
                    .
                  </p>
                </div>
              </div>

              {/* Quick Math Stats Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 w-full md:w-auto">
                <div className="p-3 bg-slate-900/90 rounded-xl border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 uppercase block font-mono">Vector Norm ||x||₂</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {latestResult.norm_x.toFixed(2)}
                  </span>
                </div>

                <div className="p-3 bg-slate-900/90 rounded-xl border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 uppercase block font-mono">Projection Residual ε</span>
                  <span className="text-sm font-mono font-bold text-violet-400">
                    {latestResult.reconstruction_error.toFixed(4)}
                  </span>
                </div>

                <div className="p-3 bg-slate-900/90 rounded-xl border border-slate-800 text-center col-span-2 sm:col-span-1">
                  <span className="text-[10px] text-slate-500 uppercase block font-mono">Variance Captured</span>
                  <span className="text-sm font-mono font-bold text-cyan-400">
                    {(latestResult.cumulative_variance_explained * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Link to deep dashboard */}
            <div className="mt-6 pt-4 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setActiveTab('dashboard')}
                className="flex items-center gap-1.5 text-xs font-semibold text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                <span>Inspect Full Vector & Eigenspace Decomposition</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Visual Visualizations Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SimilarityChart
              similarities={latestResult.similarities}
              predictedDigit={latestResult.predicted_digit}
              topCandidates={latestResult.top_candidates}
            />

            <ProjectionPlot
              projectedZ2D={latestResult.projected_z_2d}
              predictedDigit={latestResult.predicted_digit}
              kComponents={latestResult.k_components}
            />
          </div>

          {/* Subspace Reconstruction Component */}
          <ReconstructionView
            originalMatrix={latestResult.matrix_8x8}
            reconstructedMatrix={latestResult.reconstructed_matrix_8x8}
            kComponents={latestResult.k_components}
            reconstructionError={latestResult.reconstruction_error}
            cumulativeVariance={latestResult.cumulative_variance_explained}
            idempotencyError={latestResult.idempotency_error}
          />
        </div>
      )}
    </div>
  );
};
