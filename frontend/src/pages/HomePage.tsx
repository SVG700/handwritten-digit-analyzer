import React from 'react';
import {
  ArrowRight,
  ShieldCheck,
  Cpu,
  BarChart3,
  Layers,
  Sparkles,
  GitBranch,
  FileText,
} from 'lucide-react';

interface HomePageProps {
  setActiveTab: (tab: string) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ setActiveTab }) => {
  const stages = [
    {
      num: 1,
      title: 'Matrix Representation',
      formula: 'M ∈ ℝ⁸ˣ⁸',
      desc: 'Input image converted into an 8×8 grayscale matrix of pixel intensities in [0, 16].',
    },
    {
      num: 2,
      title: 'Vectorization & Euclidean Normalization',
      formula: 'x = vec(M) ∈ ℝ⁶⁴,  v = x / ||x||₂',
      desc: 'Flattened row-wise into 64 dimensions and normalized onto the unit hypersphere with zero-vector safety.',
    },
    {
      num: 3,
      title: 'Covariance & Spectral SVD',
      formula: 'C = X_cᵀ X_c / (N-1) = V Λ Vᵀ',
      desc: 'Unsupervised feature extraction strictly fitted on training data to maximize orthogonal variance.',
    },
    {
      num: 4,
      title: 'Orthogonal Projection',
      formula: 'z = W_kᵀ (v - μ) ∈ ℝᵏ,  P_k = W_k W_kᵀ',
      desc: 'Projected into top-k principal subspace. Verified idempotency P² = P and symmetry Pᵀ = P.',
    },
    {
      num: 5,
      title: 'Inner-Product Similarity',
      formula: 'sim(z, p_c) = ⟨z, p_c⟩ / (||z|| ||p_c||)',
      desc: 'Normalized inner product (cosine similarity) compared against 10 precomputed class prototypes.',
    },
    {
      num: 6,
      title: 'Argmax Classification',
      formula: 'ŷ = argmax_c sim(z, p_c)',
      desc: 'Optimal class decision with full mathematical transparency, residual error analysis, and nearest centroids.',
    },
  ];

  return (
    <div className="space-y-12 max-w-6xl mx-auto py-6">
      {/* Hero Section */}
      <section className="text-center relative py-12 px-6 rounded-3xl bg-gradient-to-b from-slate-900/80 to-[#0f172a] border border-slate-800 shadow-2xl overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-violet-600/15 rounded-full blur-3xl pointer-events-none" />

        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-semibold text-slate-300 mb-6 shadow-sm">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          Linear Algebra & Vector Calculus (MAT2303) • CA3 Project
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight max-w-4xl mx-auto leading-tight">
          Handwritten Digit Feature &{' '}
          <span className="bg-gradient-to-r from-violet-400 via-cyan-400 to-cyan-300 bg-clip-text text-transparent">
            Similarity Analyzer
          </span>
        </h1>

        <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          A rigorous mathematical system combining <strong>matrix vectorization</strong>,{' '}
          <strong>Euclidean normalization</strong>, <strong>PCA/SVD spectral decomposition</strong>,{' '}
          and <strong>cosine inner products</strong> for handwritten digit classification.
        </p>

        {/* Action buttons */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <button
            onClick={() => setActiveTab('analyzer')}
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-lg shadow-violet-600/25 transition-all transform hover:-translate-y-0.5 cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            <span>Launch Interactive Analyzer</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => setActiveTab('methodology')}
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold text-sm transition-all cursor-pointer"
          >
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Linear Algebra Methodology</span>
          </button>

          <button
            onClick={() => setActiveTab('docs')}
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold text-sm transition-all cursor-pointer"
          >
            <FileText className="w-4 h-4 text-violet-400" />
            <span>Viva Q&A & Academic Report</span>
          </button>
        </div>

        {/* Student metadata banner */}
        <div className="mt-10 pt-6 border-t border-slate-800/80 flex flex-wrap items-center justify-center gap-6 text-xs text-slate-400">
          <div>
            Student: <strong className="text-white">Samhith V Gupta</strong>
          </div>
          <div>•</div>
          <div>
            Roll No: <span className="font-mono text-violet-400">[ROLL_NO_PLACEHOLDER]</span>
          </div>
          <div>•</div>
          <div>
            Dataset: <strong className="text-cyan-400">scikit-learn 8×8 Digits</strong>
          </div>
          <div>•</div>
          <div>
            Evaluation: <strong className="text-emerald-400">Held-out Test Split</strong>
          </div>
        </div>
      </section>

      {/* 6-Stage Pipeline Overview */}
      <section className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white">The 6-Stage Mathematical Pipeline</h2>
          <p className="text-sm text-slate-400 mt-1">
            Each stage rigorously uses the previous stage's mathematical output with no fabricated values.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {stages.map((stage) => (
            <div
              key={stage.num}
              className="bg-[#0f172a] border border-slate-800 hover:border-violet-500/50 rounded-2xl p-5 shadow-lg transition-all hover:-translate-y-1 group"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="w-8 h-8 rounded-lg bg-violet-950 border border-violet-800 text-violet-300 font-mono text-xs font-bold flex items-center justify-center group-hover:bg-violet-600 group-hover:text-white transition-colors">
                  0{stage.num}
                </span>
                <span className="font-mono text-[11px] text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
                  {stage.formula}
                </span>
              </div>
              <h3 className="text-base font-semibold text-white mb-2">{stage.title}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{stage.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Dataset & Academic Honesty Guarantees */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <span>Academic Transparency & Dataset Specifications</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-300">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <h4 className="font-semibold text-white mb-1.5 flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-cyan-400" />
              8×8 Matrix Preprocessing
            </h4>
            <p className="text-slate-400 leading-relaxed">
              Based on scikit-learn's standardized 8×8 handwritten digits (1,797 total instances, 64 features).
              All canvas inputs and uploaded images are properly centered, normalized, and scaled to the authentic [0, 16] intensity range.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <h4 className="font-semibold text-white mb-1.5 flex items-center gap-1.5">
              <GitBranch className="w-4 h-4 text-violet-400" />
              Strict Train/Test Isolation
            </h4>
            <p className="text-slate-400 leading-relaxed">
              No data leakage: Covariance matrix C, eigenvectors V, mean vector μ, and class prototypes p_c are
              fitted strictly on the 80% training split. Testing and live inference reuse the fitted model.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <h4 className="font-semibold text-white mb-1.5 flex items-center gap-1.5">
              <BarChart3 className="w-4 h-4 text-amber-400" />
              Genuine Linear Algebra Results
            </h4>
            <p className="text-slate-400 leading-relaxed">
              Zero fabricated outputs: Every similarity score, projection coordinate, residual reconstruction error,
              and confusion matrix value comes directly from verified NumPy and SciPy matrix algebra.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
