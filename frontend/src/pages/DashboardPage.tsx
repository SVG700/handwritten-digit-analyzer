import React, { useState } from 'react';
import {
  Layers,
  Cpu,
  BarChart2,
  Activity,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import type { MathPipelineResponse } from '../types';
import { MatrixHeatmap } from '../components/MatrixHeatmap';

interface DashboardPageProps {
  latestResult: MathPipelineResponse | null;
  setActiveTab: (tab: string) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  latestResult,
  setActiveTab,
}) => {
  const [activeVectorTab, setActiveVectorTab] = useState<'x' | 'v' | 'vc' | 'z'>('z');

  if (!latestResult) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center">
        <div className="p-4 rounded-full bg-slate-900 border border-slate-800 text-cyan-400 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
          <Layers className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-white mb-2">No Digit Analyzed Yet</h3>
        <p className="text-sm text-slate-400 mb-6 max-w-md mx-auto">
          Draw or upload a digit on the Analyzer page to view its step-by-step mathematical decomposition.
        </p>
        <button
          onClick={() => setActiveTab('analyzer')}
          className="px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-medium text-xs transition-colors"
        >
          Go to Digit Analyzer
        </button>
      </div>
    );
  }

  const {
    matrix_8x8,
    vector_x,
    normalized_vector_v,
    centered_vector_vc,
    projected_z,
    k_components,
    norm_x,
    reconstruction_error,
    idempotency_error,
    cumulative_variance_explained,
    predicted_digit,
    similarities,
    euclidean_distances,
  } = latestResult;

  // Active vector display
  const currentVector =
    activeVectorTab === 'x'
      ? vector_x
      : activeVectorTab === 'v'
      ? normalized_vector_v
      : activeVectorTab === 'vc'
      ? centered_vector_vc
      : projected_z;

  const currentVectorLabel =
    activeVectorTab === 'x'
      ? 'Raw Flattened Vector x ∈ ℝ⁶⁴'
      : activeVectorTab === 'v'
      ? 'Euclidean Normalized Vector v = x / ||x||₂ ∈ ℝ⁶⁴'
      : activeVectorTab === 'vc'
      ? 'Mean-Centered Vector v_c = v - μ ∈ ℝ⁶⁴'
      : `Subspace Coordinates z = W_kᵀ v_c ∈ ℝ^${k_components}`;

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-4">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
            <span>Mathematical Analysis Dashboard</span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-violet-950 border border-violet-800 text-violet-300">
              MAT2303 Pipeline Trace
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Inspection of matrix representation, vector spaces, orthogonal projection, and prototype distances.
          </p>
        </div>

        <button
          onClick={() => setActiveTab('analyzer')}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-colors"
        >
          <span>Draw Another Digit</span>
          <ArrowRight className="w-3.5 h-3.5 text-cyan-400" />
        </button>
      </div>

      {/* Overview Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 shadow-md">
          <span className="text-[11px] text-slate-400 uppercase font-mono block mb-1">Euclidean Norm</span>
          <span className="text-xl font-bold font-mono text-cyan-400">
            ||x||₂ = {norm_x.toFixed(3)}
          </span>
          <span className="text-[10px] text-slate-500 block mt-1">Non-zero check: {norm_x > 0 ? 'Passed' : 'Zero Vector'}</span>
        </div>

        <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 shadow-md">
          <span className="text-[11px] text-slate-400 uppercase font-mono block mb-1">Subspace Dimension</span>
          <span className="text-xl font-bold font-mono text-violet-400">
            k = {k_components} / 64
          </span>
          <span className="text-[10px] text-slate-500 block mt-1">Variance: {(cumulative_variance_explained * 100).toFixed(1)}%</span>
        </div>

        <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 shadow-md">
          <span className="text-[11px] text-slate-400 uppercase font-mono block mb-1">Reconstruction Error</span>
          <span className="text-xl font-bold font-mono text-amber-400">
            ε = {reconstruction_error.toFixed(4)}
          </span>
          <span className="text-[10px] text-slate-500 block mt-1">||v_c - W_k z||₂</span>
        </div>

        <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 shadow-md">
          <span className="text-[11px] text-slate-400 uppercase font-mono block mb-1">Idempotency Verify</span>
          <span className="text-xl font-bold font-mono text-emerald-400">
            P² = P
          </span>
          <span className="text-[10px] text-slate-500 block mt-1">Max error: {idempotency_error.toExponential(2)}</span>
        </div>
      </div>

      {/* Stage Breakdown Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stage 1: Matrix Heatmap */}
        <div className="space-y-4">
          <MatrixHeatmap
            matrix={matrix_8x8}
            title="Stage 1: Grayscale Matrix M"
            subtitle="M ∈ ℝ⁸ˣ⁸ with pixel values in [0, 16]"
            showValuesDefault={true}
          />

          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 text-xs text-slate-300">
            <h5 className="font-semibold text-white mb-1 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              Sparsity & Dynamic Range
            </h5>
            <p className="text-slate-400 leading-relaxed">
              Total pixels: 64. Active pixels:{' '}
              {vector_x.filter((val) => val > 0).length} of 64. Maximum intensity:{' '}
              {Math.max(...vector_x).toFixed(1)} / 16.0.
            </p>
          </div>
        </div>

        {/* Vector Stream Viewer */}
        <div className="lg:col-span-2 bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
              <div>
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Activity className="w-4 h-4 text-violet-400" />
                  <span>Vector Coordinate Stream</span>
                </h4>
                <p className="text-[11px] text-slate-400">{currentVectorLabel}</p>
              </div>

              {/* Vector tabs */}
              <div className="flex items-center gap-1 p-1 bg-slate-900 rounded-xl border border-slate-800 text-xs">
                {(['x', 'v', 'vc', 'z'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveVectorTab(tab)}
                    className={`px-2.5 py-1 rounded-lg font-mono font-medium transition-all ${
                      activeVectorTab === tab
                        ? 'bg-violet-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {tab === 'x' ? 'x (Raw)' : tab === 'v' ? 'v (Unit)' : tab === 'vc' ? 'v_c' : 'z (k-D)'}
                  </button>
                ))}
              </div>
            </div>

            {/* Numerical Grid Stream */}
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 max-h-[220px] overflow-y-auto">
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5 text-center font-mono text-xs">
                {currentVector.map((val, idx) => (
                  <div
                    key={idx}
                    className="p-1.5 rounded bg-slate-900/80 border border-slate-800/80 flex flex-col"
                  >
                    <span className="text-[9px] text-slate-500">[{idx}]</span>
                    <span
                      className={`font-semibold ${
                        val > 0
                          ? 'text-cyan-400'
                          : val < 0
                          ? 'text-violet-400'
                          : 'text-slate-600'
                      }`}
                    >
                      {val.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
            <span>Dimension: {currentVector.length} elements</span>
            <span>
              Euclidean norm of stream: {Math.sqrt(currentVector.reduce((acc, c) => acc + c * c, 0)).toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Class Prototypes Comparison Table */}
      <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 shadow-lg">
        <h4 className="text-base font-bold text-white mb-2 flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-cyan-400" />
          <span>Stage 5 & 6: Prototype Similarities & Decision Table</span>
        </h4>
        <p className="text-xs text-slate-400 mb-6">
          Comparing the projected vector z against all 10 training class prototypes p_c using Cosine Similarity and Euclidean Distance.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Digit Class</th>
                <th className="py-3 px-4">Cosine Similarity sim(z, p_c)</th>
                <th className="py-3 px-4">Euclidean Distance ||z - p_c||₂</th>
                <th className="py-3 px-4">Mathematical Interpretation</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((c) => {
                const sim = similarities[String(c)] ?? 0;
                const dist = euclidean_distances[String(c)] ?? 0;
                const isWinner = c === predicted_digit;

                return (
                  <tr
                    key={c}
                    className={`transition-colors ${
                      isWinner
                        ? 'bg-violet-950/40 text-cyan-300 font-bold'
                        : 'hover:bg-slate-900/40 text-slate-300'
                    }`}
                  >
                    <td className="py-3 px-4 flex items-center gap-2 font-sans font-semibold">
                      <span
                        className={`w-6 h-6 rounded-md flex items-center justify-center text-xs ${
                          isWinner ? 'bg-cyan-500 text-black' : 'bg-slate-800 text-slate-300'
                        }`}
                      >
                        {c}
                      </span>
                      <span>Class {c}</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={isWinner ? 'text-cyan-400 font-bold' : ''}>
                        {sim >= 0 ? '+' : ''}{sim.toFixed(5)}
                      </span>
                    </td>
                    <td className="py-3 px-4">{dist.toFixed(5)}</td>
                    <td className="py-3 px-4 font-sans text-slate-400">
                      {isWinner ? (
                        <span className="text-emerald-400 font-medium">
                          Strongest collinear alignment (Argmax)
                        </span>
                      ) : sim > 0.5 ? (
                        'High collinearity'
                      ) : sim > 0 ? (
                        'Moderate correlation'
                      ) : (
                        'Orthogonal or divergent'
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {isWinner ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-950 border border-emerald-800 text-emerald-300 text-[11px] font-sans">
                          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                          Predicted
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
