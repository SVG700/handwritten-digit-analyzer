import React, { useRef, useState, useEffect } from 'react';
import { Eraser, Sparkles, Upload, RefreshCw, Sliders } from 'lucide-react';
import { fetchSampleDigit } from '../services/api';

interface CanvasDrawerProps {
  onAnalyzeCanvas: (base64: string, kComponents: number) => void;
  onAnalyzeUpload: (file: File, kComponents: number) => void;
  onAnalyzeMatrix: (matrix: number[][], kComponents: number) => void;
  isLoading: boolean;
  kComponents: number;
  setKComponents: (k: number) => void;
}

export const CanvasDrawer: React.FC<CanvasDrawerProps> = ({
  onAnalyzeCanvas,
  onAnalyzeUpload,
  onAnalyzeMatrix,
  isLoading,
  kComponents,
  setKComponents,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(18);
  const [inputMode, setInputMode] = useState<'canvas' | 'upload'>('canvas');
  const [dragActive, setDragActive] = useState(false);
  const [sampleLoading, setSampleLoading] = useState<number | null>(null);

  // Initialize canvas with black background
  const initCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  useEffect(() => {
    initCanvas();
  }, [inputMode]);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    setIsDrawing(true);
    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const handleClear = () => {
    initCanvas();
  };

  const handleTriggerAnalyze = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const base64 = canvas.toDataURL('image/png');
    onAnalyzeCanvas(base64, kComponents);
  };

  const handleFileUpload = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (PNG, JPG, etc.)');
      return;
    }
    onAnalyzeUpload(file, kComponents);
  };

  const handleLoadSample = async (digit: number) => {
    try {
      setSampleLoading(digit);
      const sample = await fetchSampleDigit(digit);
      // Pass directly to analyze matrix
      onAnalyzeMatrix(sample.matrix_8x8, kComponents);

      // Also render it onto the canvas for visual feedback
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          const cellSize = canvas.width / 8;
          for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
              const val = sample.matrix_8x8[r][c]; // 0 to 16
              const gray = Math.round((val / 16) * 255);
              ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
              ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSampleLoading(null);
    }
  };

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
      {/* Background glow accent */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-violet-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Mode selection & Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>Digit Input & Feature Capture</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Draw a single digit (0–9), upload an image, or load a sample from the test set.
          </p>
        </div>

        {/* Tab switch */}
        <div className="flex items-center p-1 bg-slate-900 border border-slate-800 rounded-xl">
          <button
            onClick={() => setInputMode('canvas')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              inputMode === 'canvas'
                ? 'bg-violet-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Draw Digit
          </button>
          <button
            onClick={() => setInputMode('upload')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              inputMode === 'upload'
                ? 'bg-violet-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Upload File
          </button>
        </div>
      </div>

      {/* Main interactive area */}
      <div className="flex flex-col lg:flex-row items-center gap-8 justify-center">
        {inputMode === 'canvas' ? (
          <div className="flex flex-col items-center gap-4">
            <div className="relative p-2 bg-slate-900 rounded-2xl border-2 border-slate-700/80 shadow-inner group">
              <canvas
                ref={canvasRef}
                width={256}
                height={256}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
                className="rounded-xl cursor-crosshair bg-black touch-none block"
              />
              <div className="absolute bottom-4 right-4 pointer-events-none text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-900/80 border border-slate-700 text-slate-400">
                256×256 Canvas
              </div>
            </div>

            {/* Canvas controls */}
            <div className="flex items-center gap-3 w-full max-w-[256px]">
              <button
                onClick={handleClear}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs font-medium transition-colors"
              >
                <Eraser className="w-3.5 h-3.5" />
                Clear
              </button>

              <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800 flex-1">
                <span className="text-[11px] text-slate-400 whitespace-nowrap">Brush:</span>
                <input
                  type="range"
                  min={10}
                  max={30}
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-full accent-cyan-400 h-1.5 bg-slate-700 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>
        ) : (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files?.[0]) {
                handleFileUpload(e.dataTransfer.files[0]);
              }
            }}
            className={`w-[272px] h-[272px] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-6 text-center transition-all ${
              dragActive
                ? 'border-cyan-400 bg-cyan-950/20'
                : 'border-slate-700 bg-slate-900/60 hover:border-violet-500/60'
            }`}
          >
            <div className="p-4 rounded-full bg-slate-800 mb-3 text-cyan-400">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-white mb-1">Drag & drop digit image</p>
            <p className="text-xs text-slate-400 mb-4">PNG, JPG, WEBP, or BMP</p>
            <label className="cursor-pointer px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium transition-all shadow-md">
              Browse Files
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
            </label>
          </div>
        )}

        {/* Configuration & Controls */}
        <div className="flex-1 flex flex-col gap-5 w-full">
          {/* PCA Principal Components slider */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-violet-400" />
                Principal Components (k):
              </span>
              <span className="font-mono text-sm font-bold text-cyan-400 bg-cyan-950/80 px-2.5 py-0.5 rounded border border-cyan-800/60">
                k = {kComponents}
              </span>
            </div>
            <input
              type="range"
              min={2}
              max={64}
              value={kComponents}
              onChange={(e) => setKComponents(Number(e.target.value))}
              className="w-full accent-violet-500 h-2 bg-slate-800 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-500 mt-1 font-mono">
              <span>k = 2 (2D plane)</span>
              <span>k = 16 (Optimal)</span>
              <span>k = 64 (Full Rank)</span>
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Controls the dimension of the subspace <span className="font-mono text-violet-300">W_k ∈ ℝ^(64×k)</span>.
            </p>
          </div>

          {/* Quick test with real test dataset samples */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
            <span className="text-xs font-semibold text-slate-300 block mb-2">
              Or test with real held-out test digits:
            </span>
            <div className="grid grid-cols-5 sm:grid-cols-10 gap-1.5">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
                <button
                  key={d}
                  onClick={() => handleLoadSample(d)}
                  disabled={sampleLoading === d || isLoading}
                  className="px-2.5 py-2 rounded-lg bg-slate-800 hover:bg-violet-600/30 hover:border-violet-500 text-slate-200 font-mono text-sm font-bold border border-slate-700 transition-all disabled:opacity-50"
                >
                  {sampleLoading === d ? '...' : d}
                </button>
              ))}
            </div>
          </div>

          {/* Primary Action Button */}
          {inputMode === 'canvas' && (
            <button
              onClick={handleTriggerAnalyze}
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-lg shadow-violet-600/25 transition-all transform active:scale-98 disabled:opacity-60 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Linear Algebra Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute 6-Stage Analysis Pipeline</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
