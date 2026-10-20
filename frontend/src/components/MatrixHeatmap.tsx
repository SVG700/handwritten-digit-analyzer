import React, { useState } from 'react';
import { Eye, Hash } from 'lucide-react';

interface MatrixHeatmapProps {
  matrix: number[][];
  title: string;
  subtitle?: string;
  maxValue?: number;
  showValuesDefault?: boolean;
}

export const MatrixHeatmap: React.FC<MatrixHeatmapProps> = ({
  matrix,
  title,
  subtitle,
  maxValue = 16.0,
  showValuesDefault = false,
}) => {
  const [showValues, setShowValues] = useState(showValuesDefault);
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number; val: number } | null>(null);

  if (!matrix || matrix.length === 0) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 text-center text-slate-500 text-xs">
        No matrix data available.
      </div>
    );
  }

  // Get color for intensity value
  const getCellColor = (val: number) => {
    const norm = Math.max(0, Math.min(1, val / maxValue));
    if (norm === 0) return 'rgba(15, 23, 42, 0.9)'; // Slate 900
    // Interpolate from deep violet to luminous cyan
    const r = Math.round(15 + norm * (6 - 15));
    const g = Math.round(23 + norm * (182 - 23));
    const b = Math.round(42 + norm * (212 - 42));
    const alpha = 0.2 + norm * 0.8;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col items-center">
      <div className="w-full flex items-center justify-between mb-3">
        <div>
          <h4 className="text-sm font-semibold text-white">{title}</h4>
          {subtitle && <p className="text-[11px] text-slate-400">{subtitle}</p>}
        </div>

        <button
          onClick={() => setShowValues(!showValues)}
          className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors"
          title="Toggle numerical pixel values"
        >
          {showValues ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <Hash className="w-3.5 h-3.5 text-violet-400" />}
          <span>{showValues ? 'Intensities' : 'Values'}</span>
        </button>
      </div>

      {/* 8x8 Matrix Container */}
      <div className="relative p-2 bg-slate-950 rounded-xl border border-slate-800/80 shadow-inner">
        <div className="grid grid-cols-8 gap-1 w-[208px] h-[208px]">
          {matrix.map((row, r) =>
            row.map((val, c) => {
              const normVal = Math.max(0, Math.min(1, val / maxValue));
              return (
                <div
                  key={`${r}-${c}`}
                  onMouseEnter={() => setHoveredCell({ r, c, val })}
                  onMouseLeave={() => setHoveredCell(null)}
                  style={{ backgroundColor: getCellColor(val) }}
                  className="w-6 h-6 rounded-sm flex items-center justify-center transition-all cursor-pointer hover:ring-2 hover:ring-cyan-400"
                >
                  {showValues && (
                    <span
                      className={`text-[8px] font-mono select-none ${
                        normVal > 0.5 ? 'text-black font-bold' : 'text-slate-400'
                      }`}
                    >
                      {Math.round(val)}
                    </span>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Cell tooltip */}
      <div className="mt-2.5 text-xs font-mono h-4 text-center">
        {hoveredCell ? (
          <span className="text-cyan-400">
            Pixel M[{hoveredCell.r}, {hoveredCell.c}] = {hoveredCell.val.toFixed(2)}
          </span>
        ) : (
          <span className="text-slate-500 text-[11px]">Hover over pixels to inspect M[i, j]</span>
        )}
      </div>
    </div>
  );
};
