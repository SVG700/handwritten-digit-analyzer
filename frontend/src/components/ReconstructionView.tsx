import React from 'react';
import { Layers, Activity, CheckCircle2 } from 'lucide-react';
import { MatrixHeatmap } from './MatrixHeatmap';

interface ReconstructionViewProps {
  originalMatrix: number[][];
  reconstructedMatrix: number[][];
  kComponents: number;
  reconstructionError: number;
  cumulativeVariance: number;
  idempotencyError: number;
}

export const ReconstructionView: React.FC<ReconstructionViewProps> = ({
  originalMatrix,
  reconstructedMatrix,
  kComponents,
  reconstructionError,
  cumulativeVariance,
  idempotencyError,
}) => {
  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6">
        <div>
          <h4 className="text-sm font-semibold text-white flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Orthogonal Subspace Reconstruction</span>
          </h4>
          <p className="text-[11px] text-slate-400">
            Projection Operator P_k = W_k W_kᵀ, v̂_c = P_k v_c
          </p>
        </div>

        {/* Idempotency Badge */}
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-800 text-xs text-emerald-300">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Idempotent: P² = P (err: {idempotencyError.toExponential(1)})</span>
        </div>
      </div>

      {/* Side-by-side matrices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center justify-center">
        <MatrixHeatmap
          matrix={originalMatrix}
          title="Input Normalized (8×8)"
          subtitle="Original vector x ∈ ℝ⁶⁴"
        />
        <MatrixHeatmap
          matrix={reconstructedMatrix}
          title={`Reconstruction (k = ${kComponents})`}
          subtitle={`Rank-${kComponents} Subspace approximation x̂`}
        />
      </div>

      {/* Metrics Bar */}
      <div className="mt-6 pt-4 border-t border-slate-800/80 grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 block mb-0.5">Reconstruction Residual Error</span>
          <span className="text-sm font-mono font-bold text-violet-400">
            ε = {reconstructionError.toFixed(5)}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">||v_c - v̂_c||₂</span>
        </div>

        <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 block mb-0.5">Cumulative Variance Captured</span>
          <span className="text-sm font-mono font-bold text-cyan-400">
            {(cumulativeVariance * 100).toFixed(2)}%
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">∑ λ_j / ∑ λ_total</span>
        </div>

        <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 block mb-0.5">Linear Algebra Property</span>
          <span className="text-xs font-mono font-semibold text-emerald-400 flex items-center justify-center gap-1">
            <Activity className="w-3.5 h-3.5" />
            Orthogonal Projection
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Range(W_k) ⊥ Ker(P_k)</span>
        </div>
      </div>
    </div>
  );
};
