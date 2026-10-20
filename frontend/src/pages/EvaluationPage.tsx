import React, { useState, useEffect } from 'react';
import { Award, BarChart3, CheckCircle2, Sliders, RefreshCw, AlertCircle } from 'lucide-react';
import type { EvaluationResponse } from '../types';
import { fetchEvaluation } from '../services/api';

export const EvaluationPage: React.FC = () => {
  const [kComponents, setKComponents] = useState(16);
  const [evalData, setEvalData] = useState<EvaluationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadEvaluation = async (k: number) => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetchEvaluation(k);
      setEvalData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load test evaluation.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadEvaluation(kComponents);
  }, [kComponents]);

  return (
    <div className="space-y-8 max-w-6xl mx-auto py-4">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950 border border-emerald-800 text-xs font-semibold text-emerald-300 mb-2">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            Held-out Test Evaluation (No Data Leakage)
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Classification Performance & Confusion Matrix
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Empirical validation on 360 unobserved test digits across different subspace dimensions.
          </p>
        </div>

        {/* Dynamic k slider */}
        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 px-4 py-2.5 rounded-2xl">
          <Sliders className="w-4 h-4 text-violet-400" />
          <span className="text-xs text-slate-300 font-medium">Rank k:</span>
          <input
            type="range"
            min={4}
            max={64}
            step={4}
            value={kComponents}
            onChange={(e) => setKComponents(Number(e.target.value))}
            className="w-28 accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
          />
          <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800">
            k = {kComponents}
          </span>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/80 border border-red-800 text-red-200 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {isLoading && (
        <div className="p-8 text-center text-slate-400 flex items-center justify-center gap-2">
          <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
          <span>Evaluating test set for k = {kComponents}...</span>
        </div>
      )}

      {evalData && !isLoading && (
        <>
          {/* Key Metric Highlights */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg text-center">
              <span className="text-xs text-slate-400 uppercase font-mono block mb-1">Overall Accuracy</span>
              <span className="text-3xl font-black text-white font-mono">
                {(evalData.accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-[11px] text-emerald-400 block mt-1">
                {evalData.correct_predictions} / {evalData.total_test_samples} correctly identified
              </span>
            </div>

            <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg text-center">
              <span className="text-xs text-slate-400 uppercase font-mono block mb-1">Subspace Dimension</span>
              <span className="text-3xl font-black text-violet-400 font-mono">
                k = {evalData.k_components}
              </span>
              <span className="text-[11px] text-slate-500 block mt-1">out of 64 features ({(evalData.k_components / 64 * 100).toFixed(0)}% retention)</span>
            </div>

            <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg text-center">
              <span className="text-xs text-slate-400 uppercase font-mono block mb-1">Explained Variance</span>
              <span className="text-3xl font-black text-cyan-400 font-mono">
                {(evalData.cumulative_variance * 100).toFixed(1)}%
              </span>
              <span className="text-[11px] text-slate-500 block mt-1">Eigenvalue energy fraction</span>
            </div>

            <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg text-center">
              <span className="text-xs text-slate-400 uppercase font-mono block mb-1">Test Split Ratio</span>
              <span className="text-3xl font-black text-amber-400 font-mono">
                20%
              </span>
              <span className="text-[11px] text-slate-500 block mt-1">Stratified 1,437 train / 360 test</span>
            </div>
          </div>

          {/* 10x10 Confusion Matrix */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 shadow-lg">
            <h3 className="text-base font-bold text-white mb-2 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <span>10×10 Confusion Matrix (True vs. Predicted)</span>
            </h3>
            <p className="text-xs text-slate-400 mb-6">
              Row: Ground Truth Digit Class | Column: Model Prediction via Cosine Argmax. Diagonal entries indicate correct classifications.
            </p>

            <div className="overflow-x-auto flex justify-center">
              <div className="inline-block">
                {/* Column Headers */}
                <div className="flex items-center gap-1 mb-1 ml-10">
                  <span className="text-[10px] text-slate-500 uppercase font-mono mr-2">Pred →</span>
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((col) => (
                    <div key={col} className="w-8 text-center text-xs font-mono font-bold text-slate-400">
                      {col}
                    </div>
                  ))}
                </div>

                {/* Rows */}
                {evalData.confusion_matrix.map((row, r) => (
                  <div key={r} className="flex items-center gap-1 mb-1">
                    <span className="w-8 text-right text-xs font-mono font-bold text-slate-400 mr-2">
                      {r}
                    </span>
                    {row.map((count, c) => {
                      const isDiagonal = r === c;
                      const hasErrors = !isDiagonal && count > 0;
                      return (
                        <div
                          key={`${r}-${c}`}
                          className={`w-8 h-8 rounded-lg flex items-center justify-center font-mono text-xs transition-all ${
                            isDiagonal
                              ? count > 30
                                ? 'bg-cyan-500 text-black font-extrabold shadow-sm'
                                : 'bg-cyan-700 text-white font-bold'
                              : hasErrors
                              ? 'bg-red-950/80 text-red-300 font-bold border border-red-800/80'
                              : 'bg-slate-900/60 text-slate-600'
                          }`}
                          title={`True: ${r}, Pred: ${c} — Count: ${count}`}
                        >
                          {count}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Per-Class Metrics Table */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 shadow-lg">
            <h3 className="text-base font-bold text-white mb-2 flex items-center gap-2">
              <Award className="w-5 h-5 text-violet-400" />
              <span>Per-Digit Classification Metrics</span>
            </h3>
            <p className="text-xs text-slate-400 mb-6">
              Precision (TP / (TP + FP)), Recall (TP / (TP + FN)), and F1-Score for each individual digit class.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Digit</th>
                    <th className="py-3 px-4">Precision</th>
                    <th className="py-3 px-4">Recall</th>
                    <th className="py-3 px-4">F1-Score</th>
                    <th className="py-3 px-4">Test Samples (Support)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {evalData.per_class_metrics.map((m) => (
                    <tr key={m.digit} className="hover:bg-slate-900/40">
                      <td className="py-2.5 px-4 font-bold text-white flex items-center gap-2">
                        <span className="w-6 h-6 rounded bg-slate-800 flex items-center justify-center text-cyan-400">
                          {m.digit}
                        </span>
                        <span>Digit {m.digit}</span>
                      </td>
                      <td className="py-2.5 px-4 text-cyan-300">{(m.precision * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-4 text-violet-300">{(m.recall * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-4 text-emerald-400 font-bold">{(m.f1_score * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-4 text-slate-400">{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
