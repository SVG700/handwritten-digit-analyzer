import React from 'react';
import { Target } from 'lucide-react';

interface ProjectionPlotProps {
  projectedZ2D: number[]; // [z1, z2]
  predictedDigit: number;
  kComponents: number;
}

// 2D approximate centroid positions for digits 0-9 in the top-2 PCA space
// derived from training set centers for realistic orientation
const DIGIT_PROTOTYPE_2D: Record<number, { x: number; y: number; color: string }> = {
  0: { x: -0.42, y: -0.18, color: '#f87171' }, // Red-400
  1: { x: 0.35, y: 0.38, color: '#fb923c' },  // Orange-400
  2: { x: -0.05, y: -0.25, color: '#facc15' }, // Yellow-400
  3: { x: 0.12, y: -0.15, color: '#4ade80' },  // Green-400
  4: { x: -0.32, y: 0.32, color: '#38bdf8' },  // Sky-400
  5: { x: 0.18, y: -0.22, color: '#818cf8' },  // Indigo-400
  6: { x: -0.38, y: 0.05, color: '#c084fc' },  // Purple-400
  7: { x: 0.28, y: 0.25, color: '#f472b6' },   // Pink-400
  8: { x: 0.08, y: -0.05, color: '#2dd4bf' },  // Teal-400
  9: { x: 0.22, y: 0.10, color: '#fb7185' },   // Rose-400
};

export const ProjectionPlot: React.FC<ProjectionPlotProps> = ({
  projectedZ2D,
  predictedDigit,
  kComponents,
}) => {
  const z1 = projectedZ2D?.[0] ?? 0;
  const z2 = projectedZ2D?.[1] ?? 0;

  // Coordinate mapping from math range [-0.8, 0.8] to SVG [20, 280]
  const mapX = (val: number) => {
    const clamped = Math.max(-0.75, Math.min(0.75, val));
    return 150 + (clamped / 0.75) * 120;
  };

  const mapY = (val: number) => {
    const clamped = Math.max(-0.75, Math.min(0.75, val));
    // Invert Y for screen coordinates
    return 150 - (clamped / 0.75) * 120;
  };

  const inputX = mapX(z1);
  const inputY = mapY(z2);

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col items-center">
      <div className="w-full flex items-center justify-between mb-3">
        <div>
          <h4 className="text-sm font-semibold text-white flex items-center gap-1.5">
            <Target className="w-4 h-4 text-cyan-400" />
            <span>2D Eigenspace Projection</span>
          </h4>
          <p className="text-[11px] text-slate-400">
            Coordinates z = (u₁ᵀ(v-μ), u₂ᵀ(v-μ)) in Principal Subspace
          </p>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-violet-950/80 text-violet-300 border border-violet-800">
          Rank {kComponents} Subspace
        </span>
      </div>

      {/* SVG Scatter Plot */}
      <div className="relative bg-slate-950 rounded-xl p-2 border border-slate-800/80 w-[300px] h-[300px]">
        <svg viewBox="0 0 300 300" className="w-full h-full overflow-visible">
          {/* Grid lines */}
          <line x1="30" y1="150" x2="270" y2="150" stroke="#334155" strokeDasharray="3 3" />
          <line x1="150" y1="30" x2="150" y2="270" stroke="#334155" strokeDasharray="3 3" />

          {/* Axis Labels */}
          <text x="260" y="145" fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">
            PC1 (u₁)
          </text>
          <text x="155" y="40" fill="#64748b" fontSize="9" fontFamily="monospace">
            PC2 (u₂)
          </text>

          {/* Prototype Centroids */}
          {Object.entries(DIGIT_PROTOTYPE_2D).map(([digitStr, proto]) => {
            const d = Number(digitStr);
            const px = mapX(proto.x);
            const py = mapY(proto.y);
            const isWinner = d === predictedDigit;

            return (
              <g key={d} className="transition-all">
                {/* Connecting line from input to winner */}
                {isWinner && (
                  <line
                    x1={inputX}
                    y1={inputY}
                    x2={px}
                    y2={py}
                    stroke="#22d3ee"
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                )}
                {/* Prototype circle */}
                <circle
                  cx={px}
                  cy={py}
                  r={isWinner ? 12 : 9}
                  fill={isWinner ? '#22d3ee' : proto.color}
                  fillOpacity={isWinner ? 0.35 : 0.2}
                  stroke={isWinner ? '#22d3ee' : proto.color}
                  strokeWidth={isWinner ? 2.5 : 1.5}
                />
                <text
                  x={px}
                  y={py + 3.5}
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize="10"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  {d}
                </text>
              </g>
            );
          })}

          {/* Current Input Point */}
          <g>
            {/* Pulse animation ring */}
            <circle
              cx={inputX}
              cy={inputY}
              r="14"
              fill="none"
              stroke="#a855f7"
              strokeWidth="2"
              className="animate-ping opacity-75 origin-center"
            />
            {/* Center dot */}
            <circle
              cx={inputX}
              cy={inputY}
              r="6"
              fill="#c084fc"
              stroke="#ffffff"
              strokeWidth="2"
            />
          </g>
        </svg>

        {/* Legend */}
        <div className="absolute top-3 left-3 bg-slate-900/90 border border-slate-800 rounded px-2 py-1 text-[10px] text-slate-300 font-mono">
          Input: ({z1.toFixed(3)}, {z2.toFixed(3)})
        </div>
      </div>

      <div className="w-full mt-3 flex items-center justify-between text-[11px] text-slate-400">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-violet-400 inline-block" /> Input Point
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" /> Nearest Prototype (Digit {predictedDigit})
        </span>
      </div>
    </div>
  );
};
