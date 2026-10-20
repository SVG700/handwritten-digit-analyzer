import React from 'react';
import { Award, Compass } from 'lucide-react';
import type { CandidateScore } from '../types';

interface SimilarityChartProps {
  similarities: Record<string, number>;
  predictedDigit: number;
  topCandidates: CandidateScore[];
}

export const SimilarityChart: React.FC<SimilarityChartProps> = ({
  similarities,
  predictedDigit,
  topCandidates,
}) => {
  const digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  // Map to sorted list
  const maxSim = Math.max(...Object.values(similarities || { 0: 1 }));

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-white flex items-center gap-1.5">
            <Compass className="w-4 h-4 text-violet-400" />
            <span>Cosine Similarity Distribution (0–9)</span>
          </h4>
          <p className="text-[11px] text-slate-400">
            sim(z, p_c) = ⟨z, p_c⟩ / (||z||·||p_c||) ∈ [-1, 1]
          </p>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-violet-950/80 border border-violet-800 text-xs text-violet-300">
          <Award className="w-3.5 h-3.5 text-cyan-400" />
          <span>Winner: Digit <strong>{predictedDigit}</strong></span>
        </div>
      </div>

      {/* Bar Chart Grid */}
      <div className="space-y-2">
        {digits.map((d) => {
          const sim = similarities?.[String(d)] ?? 0;
          const isWinner = d === predictedDigit;
          // Normalize to [0, 100]% for bar length
          const barPct = Math.max(0, Math.min(100, (sim / (maxSim > 0 ? maxSim : 1)) * 100));

          return (
            <div key={d} className="flex items-center gap-3">
              {/* Digit Label */}
              <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center font-mono font-bold text-xs transition-all ${
                  isWinner
                    ? 'bg-gradient-to-br from-violet-600 to-cyan-500 text-white shadow-md shadow-cyan-500/30'
                    : 'bg-slate-800 text-slate-400 border border-slate-700/60'
                }`}
              >
                {d}
              </div>

              {/* Bar Track */}
              <div className="flex-1 h-5 bg-slate-900 rounded-lg p-0.5 border border-slate-800 relative overflow-hidden">
                <div
                  style={{ width: `${barPct}%` }}
                  className={`h-full rounded-md transition-all duration-500 ${
                    isWinner
                      ? 'bg-gradient-to-r from-violet-600 via-cyan-500 to-cyan-400 shadow-sm'
                      : 'bg-slate-700/80 hover:bg-slate-600'
                  }`}
                />
              </div>

              {/* Numeric Value */}
              <div className="w-16 text-right font-mono text-xs">
                <span
                  className={
                    isWinner ? 'text-cyan-400 font-bold' : 'text-slate-400'
                  }
                >
                  {sim >= 0 ? '+' : ''}{sim.toFixed(4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Top 3 Rankings Box */}
      <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">Rankings:</span>
        <div className="flex items-center gap-4">
          {topCandidates.slice(0, 3).map((cand) => (
            <div key={cand.digit} className="flex items-center gap-1.5 font-mono">
              <span className="text-slate-500">#{cand.rank}</span>
              <span className="text-white font-bold">Digit {cand.digit}</span>
              <span className="text-cyan-400">({cand.similarity.toFixed(3)})</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
