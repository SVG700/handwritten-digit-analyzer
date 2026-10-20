import React from 'react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="space-y-10 max-w-5xl mx-auto py-4 text-slate-300">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-950 border border-violet-800 text-xs font-semibold text-violet-300 mb-3">
          MAT2303 Mathematical Rigor
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Linear Algebra & Vector Calculus Methodology
        </h2>
        <p className="text-sm text-slate-400 mt-2 leading-relaxed">
          Detailed mathematical formulation of all 6 sequential stages implemented in the analyzer.
        </p>
      </div>

      {/* Stage 1 */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-violet-600 text-white font-mono text-xs flex items-center justify-center">1</span>
            <span>Matrix Representation of 8×8 Digit Images</span>
          </h3>
          <span className="font-mono text-xs text-cyan-400">M ∈ ℝ⁸ˣ⁸</span>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          An input handwritten digit image is modeled as a real-valued matrix:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center">
          M = [ m_(i, j) ]_(8×8) ,  where m_(i, j) ∈ [0.0, 16.0]
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          The intensity range <code className="text-violet-300">[0, 16]</code> represents grayscale reflectance,
          where 0 corresponds to background and 16 corresponds to maximum stroke saturation, matching the NIST 8×8 digit standard.
        </p>
      </section>

      {/* Stage 2 */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-violet-600 text-white font-mono text-xs flex items-center justify-center">2</span>
            <span>Vectorization & Euclidean Normalization</span>
          </h3>
          <span className="font-mono text-xs text-cyan-400">v ∈ S⁶³ ⊂ ℝ⁶⁴</span>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          The vectorization operator <code className="text-cyan-300">vec: ℝ⁸ˣ⁸ → ℝ⁶⁴</code> maps the matrix into a single column vector by stacking its rows:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center space-y-2">
          <div>x = vec(M) = [ m_(0,0), m_(0,1), ..., m_(7,7) ]ᵀ ∈ ℝ⁶⁴</div>
          <div className="text-slate-400">||x||₂ = √( ∑_(i=1)⁶⁴ x_i² )</div>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          To remove scale variance caused by differing stroke thickness or pen pressure, we project onto the unit hypersphere:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center">
          v = {'{'} x / ||x||₂  if ||x||₂ &gt; 0;  0  if ||x||₂ = 0 {'}'}
        </div>
        <div className="p-3 bg-violet-950/40 border border-violet-800/60 rounded-xl text-xs text-violet-300">
          <strong>Zero-Vector Handling:</strong> When a blank canvas or empty image is provided, ||x||₂ = 0.
          The pipeline explicitly guards against division by zero by returning the zero vector 0 ∈ ℝ⁶⁴ rather than crashing with NaN.
        </div>
      </section>

      {/* Stage 3 */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-violet-600 text-white font-mono text-xs flex items-center justify-center">3</span>
            <span>PCA & Spectral Eigendecomposition of Sample Covariance</span>
          </h3>
          <span className="font-mono text-xs text-cyan-400">C = V Λ Vᵀ</span>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          Given a training matrix of normalized samples X_train ∈ ℝ^(N_train × 64), we calculate the empirical mean and mean-centered matrix:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center space-y-2">
          <div>μ = (1 / N_train) ∑_(i=1)^(N_train) v_i ∈ ℝ⁶⁴</div>
          <div>X_c = X_train - 1 μᵀ ∈ ℝ^(N_train × 64)</div>
          <div>C = (1 / (N_train - 1)) X_cᵀ X_c ∈ ℝ^(64×64)</div>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          By the <strong>Spectral Theorem</strong> for real symmetric matrices, C is symmetric and positive semi-definite (C = Cᵀ, yᵀ C y ≥ 0 for all y).
          Hence, there exists an orthonormal basis of eigenvectors:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center">
          C u_j = λ_j u_j ,  with λ₁ ≥ λ₂ ≥ ... ≥ λ₆₄ ≥ 0  and  u_iᵀ u_j = δ_(ij)
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          The explained variance ratio for top k components is given by EVR(k) = (∑_(j=1)ᵏ λ_j) / (∑_(j=1)⁶⁴ λ_j).
        </p>
      </section>

      {/* Stage 4 */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-violet-600 text-white font-mono text-xs flex items-center justify-center">4</span>
            <span>Orthogonal Subspace Projection</span>
          </h3>
          <span className="font-mono text-xs text-cyan-400">P_k² = P_k , P_kᵀ = P_k</span>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          Let W_k = [ u₁, u₂, ..., u_k ] ∈ ℝ^(64×k) form the orthonormal basis for the top-k principal subspace.
          Notice that <code className="text-cyan-300">W_kᵀ W_k = I_k</code>.
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center space-y-2">
          <div>z = W_kᵀ (v - μ) ∈ ℝᵏ  (Low-dimensional coordinates)</div>
          <div>v̂_c = W_k z = W_k W_kᵀ (v - μ) = P_k (v - μ) ∈ ℝ⁶⁴</div>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          The linear projection operator <code className="text-cyan-300">P_k = W_k W_kᵀ</code> satisfies the two fundamental axioms of an orthogonal projector:
        </p>
        <ul className="list-disc list-inside text-xs text-slate-300 space-y-1 pl-2">
          <li><strong>Idempotency:</strong> P_k² = (W_k W_kᵀ)(W_k W_kᵀ) = W_k (W_kᵀ W_k) W_kᵀ = W_k I_k W_kᵀ = P_k</li>
          <li><strong>Symmetry (Self-Adjointness):</strong> P_kᵀ = (W_k W_kᵀ)ᵀ = (W_kᵀ)ᵀ W_kᵀ = W_k W_kᵀ = P_k</li>
        </ul>
        <p className="text-xs text-slate-400 leading-relaxed">
          The residual vector (I - P_k)(v - μ) lies in the orthogonal complement Ker(P_k) = Range(W_k)ᗮ,
          and the reconstruction error is ε = ||(I - P_k)(v - μ)||₂.
        </p>
      </section>

      {/* Stage 5 & 6 */}
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-violet-600 text-white font-mono text-xs flex items-center justify-center">5 & 6</span>
            <span>Inner-Product Cosine Similarity & Argmax Decision</span>
          </h3>
          <span className="font-mono text-xs text-cyan-400">ŷ = argmax_c sim(z, p_c)</span>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          For each digit class c ∈ {'{'}0, 1, ..., 9{'}'}, we compute the class prototype from training samples belonging to c:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center">
          p_c = (1 / |S_c|) ∑_(i ∈ S_c) z_i ∈ ℝᵏ
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          The normalized inner product (Cosine Similarity) computes the cosine of the angle θ between projected feature vector z and prototype p_c:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center space-y-2">
          <div>sim(z, p_c) = cos(θ) = ⟨z, p_c⟩ / (||z||₂ ||p_c||₂)</div>
          <div className="text-slate-400">By Cauchy-Schwarz: |⟨z, p_c⟩| ≤ ||z||₂ ||p_c||₂  ⟹  sim(z, p_c) ∈ [-1.0, 1.0]</div>
        </div>
        <p className="text-sm leading-relaxed text-slate-300">
          The final classification decision chooses the digit class whose prototype maximizes cosine collinearity:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 overflow-x-auto text-center font-bold">
          ŷ = argmax_(c ∈ {'{'}0, ..., 9{'}'}) sim(z, p_c)
        </div>
      </section>
    </div>
  );
};
