import React from 'react';
import {
  FileText,
  HelpCircle,
  Code,
  Terminal,
  FileCheck,
} from 'lucide-react';

export const DocsPage: React.FC = () => {
  const vivaQuestions = [
    {
      q: '1. Why use Cosine Similarity instead of Euclidean Distance alone for digit classification?',
      a: 'Cosine similarity measures directional collinearity in eigenspace (cos θ = ⟨z, p_c⟩ / (||z|| ||p_c||)). Even if a digit is drawn with lighter pressure or partial gaps, its dominant directional features remain aligned with the true class prototype. Euclidean distance can penalize uniform intensity variations heavily.',
    },
    {
      q: '2. What is the geometric meaning of the projection operator P_k = W_k W_kᵀ?',
      a: 'P_k acts as an orthogonal projector that decomposes any vector into two mutually orthogonal components: v_c = P_k v_c + (I - P_k) v_c, where P_k v_c lies in the principal eigenspace (Range(W_k)) and (I - P_k) v_c lies in the orthogonal complement (null space of P_k), which contains high-frequency noise.',
    },
    {
      q: '3. Why must an orthogonal projection matrix satisfy P² = P and Pᵀ = P?',
      a: 'Idempotency (P² = P) signifies that once a vector is orthogonally projected into the subspace, a second projection leaves it invariant. Symmetry (Pᵀ = P) ensures that the projector is self-adjoint, guaranteeing that the null space and range are orthogonal to each other.',
    },
    {
      q: '4. Why is the scikit-learn 8×8 dataset used instead of 28×28 MNIST?',
      a: 'The 8×8 format gives an exact 64-dimensional vector space. In this space, the 64×64 covariance matrix can be fully decomposed via classical spectral eigendecomposition in closed form. This allows full mathematical transparency for MAT2303 viva examination without requiring convolutional neural networks or black-box classifiers.',
    },
    {
      q: '5. How is data leakage rigorously prevented during the pipeline execution?',
      a: 'All preprocessing parameters (the mean vector μ, the covariance matrix C, the eigenvector basis W_k, and the class prototypes p_c) are computed exclusively on the 80% training set. The 20% test set samples are held out and evaluated only using the previously frozen parameters.',
    },
    {
      q: '6. How does the system handle zero vectors or completely blank drawings?',
      a: 'If a blank canvas is analyzed, ||x||₂ = 0. A standard division x / ||x||₂ produces division-by-zero errors (NaN/Inf). The pipeline checks for ||x||₂ < 10⁻¹² and explicitly returns the zero vector 0 ∈ ℝ⁶⁴ and flags is_blank = true, ensuring robust production resilience.',
    },
  ];

  return (
    <div className="space-y-10 max-w-5xl mx-auto py-4 text-slate-300">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-950 border border-violet-800 text-xs font-semibold text-violet-300 mb-3">
          MAT2303 Academic Submission
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Project Documentation & Viva Voce Guide
        </h2>
        <p className="text-sm text-slate-400 mt-2 leading-relaxed">
          Comprehensive review for mathematics project presentation, viva examination, and submission artifacts.
        </p>
      </div>

      {/* Submission Deliverables */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <FileCheck className="w-5 h-5 text-emerald-400" />
          <span>Mandatory Submission Deliverables</span>
        </h3>
        <p className="text-xs text-slate-400">
          The project generates all required university submission files in the root workspace directory:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-start gap-3">
            <div className="p-2.5 rounded-lg bg-violet-950 border border-violet-800 text-violet-300 shrink-0">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-mono text-sm font-bold text-white">SamhithVGupta_RollNo_CA3.pdf</h4>
              <p className="text-xs text-slate-400 mt-1">
                Formal CA3 assignment report with mathematical proofs, flowcharts, system architecture, verified results, and references.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-start gap-3">
            <div className="p-2.5 rounded-lg bg-cyan-950 border border-cyan-800 text-cyan-300 shrink-0">
              <Code className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-mono text-sm font-bold text-white">SamhithVGupta_RollNo_Code.ipynb</h4>
              <p className="text-xs text-slate-400 mt-1">
                Fully executed, reproducible Jupyter Notebook detailing steps 1 through 13 with embedded matplotlib visualizations.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Viva Q&A */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <HelpCircle className="w-5 h-5 text-cyan-400" />
          <span>Viva Voce & Technical Defense Q&A</span>
        </h3>

        <div className="space-y-4">
          {vivaQuestions.map((item, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
              <h4 className="text-sm font-bold text-white mb-2">{item.q}</h4>
              <p className="text-xs text-slate-400 leading-relaxed">{item.a}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Terminal / Run Instructions */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Terminal className="w-5 h-5 text-amber-400" />
          <span>Execution Commands (Windows PowerShell)</span>
        </h3>

        <div className="space-y-3 font-mono text-xs">
          <div>
            <span className="text-slate-400 block mb-1">Start FastAPI Backend:</span>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-cyan-300">
              .\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --port 8000
            </div>
          </div>

          <div>
            <span className="text-slate-400 block mb-1">Start React + Vite Frontend:</span>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-cyan-300">
              cd frontend; npm run dev
            </div>
          </div>

          <div>
            <span className="text-slate-400 block mb-1">Run Automated Test Suite:</span>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-cyan-300">
              .\.venv\Scripts\pytest.exe -v
            </div>
          </div>

          <div>
            <span className="text-slate-400 block mb-1">Regenerate CA3 PDF Report:</span>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-cyan-300">
              .\.venv\Scripts\python.exe report\generate_pdf.py
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
