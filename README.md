# Handwritten Digit Feature and Similarity Analyzer
### Linear Algebra & Vector Calculus (MAT2303) — Continuous Assessment 3 (CA3) Project

**Student Name:** Samhith V Gupta  
**Roll / Registration Number:** `[ROLL_NO_PLACEHOLDER]`  
**Course Code:** MAT2303 (Linear Algebra and Vector Calculus)  
**Submission Artifacts:**
1. `SamhithVGupta_RollNo_Code.ipynb` (Executable Jupyter Notebook with 31 cells & embedded visualizations)
2. `SamhithVGupta_RollNo_CA3.pdf` (Formal 8-page Academic Assignment Report)
3. Full-Stack Web Application (React 19 + TypeScript + Vite + Tailwind CSS + FastAPI + NumPy + SciPy)

---

## 1. Project Overview & Mathematical Architecture

This project implements an **interpretable, authentic, and academically rigorous** handwritten digit recognition system based purely on **Linear Algebra and Vector Calculus**. Rather than relying on black-box neural networks, every prediction is computed through an exact **6-stage linear algebraic pipeline**:

```
[Raw Image / Drawing] 
        ↓
  1. Matrix Representation: M ∈ ℝ^(8×8), m_(i,j) ∈ [0, 16]
        ↓
  2. Vectorization & Euclidean Normalization: x = vec(M) ∈ ℝ^64, v = x / ||x||_2
        ↓
  3. Covariance & Spectral SVD: C = X_cᵀ X_c / (N-1) = V Λ Vᵀ  (Fitted strictly on Training Split)
        ↓
  4. Orthogonal Projection: z = W_kᵀ (v - μ) ∈ ℝ^k,  P_k = W_k W_kᵀ  (Idempotent: P² = P, Symmetric: Pᵀ = P)
        ↓
  5. Directional Inner-Product Similarity: sim(z, p_c) = ⟨z, p_c⟩ / (||z||_2 ||p_c||_2) ∈ [-1, 1]
        ↓
  6. Classification Decision: ŷ = argmax_(c ∈ {0..9}) sim(z, p_c)
```

### Dataset Specifications & Anti-Leakage Guarantee
- **Dataset:** Standardized scikit-learn 8×8 handwritten digits (1,797 samples, 64 features).
- **Partition:** 80% stratified training split (1,437 samples) and 20% held-out test split (360 samples).
- **Strict Isolation:** Preprocessing statistics (empirical mean $\boldsymbol{\mu}$, sample covariance $\mathbf{C}$, eigenvector basis $\mathbf{W}_k$, and class prototypes $\mathbf{p}_c$) are computed **strictly on the training split**. Unobserved test samples are evaluated using these fixed parameters without data leakage.

---

## 2. Directory Structure

```
maths project/
├── .venv/                                # Python virtual environment
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI application & REST endpoints
│   │   ├── math_pipeline.py              # Core 6-stage linear algebra engine
│   │   ├── preprocessor.py               # Canvas decoder & 8x8 downsampler
│   │   └── schemas.py                    # Pydantic API data models
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_math_pipeline.py         # Unit tests for norms, SVD, P²=P, cosine
│   │   └── test_api.py                   # FastAPI integration tests
│   └── requirements.txt
├── frontend/                             # Modern Web Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── CanvasDrawer.tsx          # HTML5 Canvas drawer + sample digit loader
│   │   │   ├── MatrixHeatmap.tsx         # 8x8 interactive pixel intensity visualizer
│   │   │   ├── ProjectionPlot.tsx        # 2D eigenspace scatter plot with prototypes
│   │   │   ├── SimilarityChart.tsx       # Cosine similarity bar chart (0-9)
│   │   │   ├── ReconstructionView.tsx    # Subspace rank-k image reconstruction
│   │   │   ├── Navbar.tsx                # Navigation header
│   │   │   └── Footer.tsx                # Metadata footer
│   │   ├── pages/
│   │   │   ├── HomePage.tsx              # Overview & 6-stage interactive cards
│   │   │   ├── AnalyzerPage.tsx          # Live digit analyzer & results
│   │   │   ├── DashboardPage.tsx         # Deep vector coordinate & matrix inspect
│   │   │   ├── MethodologyPage.tsx       # Formal proofs & mathematical derivations
│   │   │   ├── EvaluationPage.tsx        # Live 10x10 confusion matrix & metrics
│   │   │   └── DocsPage.tsx              # Viva Voce Q&A and execution guide
│   │   ├── services/
│   │   │   └── api.ts                    # Backend API client
│   │   ├── types/
│   │   │   └── index.ts                  # TypeScript interface definitions
│   │   ├── App.tsx                       # Root layout & page switcher
│   │   ├── main.tsx
│   │   └── index.css                     # Tailwind v4 dark navy theme styling
│   ├── package.json
│   ├── vite.config.ts                    # Vite config with backend proxy
│   └── tsconfig.json
├── notebook/
│   ├── generate_and_execute_notebook.py  # Automated notebook builder & executor
│   └── SamhithVGupta_RollNo_Code.ipynb   # Executed Jupyter Notebook backup
├── report/
│   ├── generate_pdf.py                   # ReportLab PDF compiler
│   ├── assets/                           # High-resolution generated charts
│   │   ├── scree_plot.png
│   │   ├── confusion_matrix.png
│   │   ├── flowchart.png
│   │   └── dim_sweep.png
│   └── SamhithVGupta_RollNo_CA3.pdf      # Formal CA3 PDF Report backup
├── SamhithVGupta_RollNo_Code.ipynb       # ROOT SUBMISSION NOTEBOOK (31 cells)
├── SamhithVGupta_RollNo_CA3.pdf          # ROOT SUBMISSION CA3 PDF (8 pages)
├── pytest.ini                            # Pytest configuration
├── requirements.txt                      # Project-wide Python dependencies
└── README.md                             # Comprehensive project documentation
```

---

## 3. Windows PowerShell Setup & Execution Guide

### Prerequisites
- Windows 10/11
- Python 3.10+ (Tested on Python 3.15.0)
- Node.js v18+ & npm (Tested on Node v24.20.0, npm 11.19.0)

### Step 1: Virtual Environment Setup
Open Windows PowerShell in the project directory:
```powershell
# Navigate to project folder
cd "C:\Users\User\Desktop\maths project"

# Activate existing virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Verify dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 2: Run the Automated Test Suite (18 Tests)
Execute the pytest suite covering vectorization, zero vectors, orthogonality, idempotency, and API endpoints:
```powershell
.\.venv\Scripts\pytest.exe -v
```
*Expected Result: 18 passed in ~15s.*

### Step 3: Launch the FastAPI Backend
Start the uvicorn API server on port 8000:
```powershell
.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --port 8000
```
- Health Check: `http://127.0.0.1:8000/api/health`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

### Step 4: Launch the React + Vite Frontend
In a second PowerShell terminal:
```powershell
cd "C:\Users\User\Desktop\maths project\frontend"
npm run dev
```
Open your browser at: **`http://localhost:5173`**

### Step 5: Regenerate Submission Artifacts (If Needed)
To re-execute the Jupyter notebook:
```powershell
.\.venv\Scripts\python.exe notebook\generate_and_execute_notebook.py
```
To re-compile the official PDF report:
```powershell
.\.venv\Scripts\python.exe report\generate_pdf.py
```

---

## 4. Key Mathematical Theorems & Properties Demonstrated

| Concept | Formal Equation | Algebraic / Geometric Property |
| :--- | :--- | :--- |
| **Vector Space Isomorphism** | $\text{vec}: \mathbb{R}^{8 \times 8} \to \mathbb{R}^{64}$ | Bijective linear mapping preserving inner products. |
| **Zero-Vector Safety** | $\mathbf{v} = \mathbf{x} / \|\mathbf{x}\|_2 \text{ if } \|\mathbf{x}\| > 0 \text{ else } \mathbf{0}$ | Prevents $\frac{0}{0}$ division by zero on blank drawings. |
| **Spectral Theorem** | $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$ | Real symmetric covariance matrix has orthonormal eigenvectors ($\mathbf{V}^T \mathbf{V} = \mathbf{I}$). |
| **Orthogonal Projector** | $P_k = \mathbf{W}_k \mathbf{W}_k^T$ | **Idempotent:** $P_k^2 = P_k$<br/>**Self-Adjoint:** $P_k^T = P_k$<br/>$\text{Ker}(P_k) \perp \text{Range}(W_k)$. |
| **Cauchy-Schwarz Inequality** | $\|\langle \mathbf{z}, \mathbf{p} \rangle\| \le \|\mathbf{z}\|_2 \|\mathbf{p}\|_2$ | Strictly bounds Cosine Similarity in $[-1.0, 1.0]$. |
| **Optimal Subspace Rank** | $k = 16$ | Preserves **75.8% variance**, achieves **88.89% test accuracy**, $4\times$ compression. |

---

## 5. Viva Voce Defense & Examination Q&A

1. **Q: Why use Cosine Similarity instead of Euclidean Distance?**
   - *A:* Cosine similarity evaluates directional collinearity on the unit hypersphere ($\cos \theta = \frac{\langle \mathbf{z}, \mathbf{p}_c \rangle}{\|\mathbf{z}\| \|\mathbf{p}_c\|}$). Even when a user writes with light pencil strokes or varying thickness, the relative coordinate distribution aligns with the class prototype. Euclidean distance can be distorted by scalar intensity shifts.

2. **Q: What is the significance of the projection residual $\epsilon = \|(I - P_k)\mathbf{v}_c\|_2$?**
   - *A:* By the Orthogonal Decomposition Theorem, any vector decomposes into $\mathbf{v}_c = P_k \mathbf{v}_c + (I - P_k)\mathbf{v}_c$. The residual $(I - P_k)\mathbf{v}_c$ lies in the null space of $P_k$, capturing high-frequency noise and non-principal variance.

3. **Q: Why was the 8×8 dataset chosen over 28×28 MNIST?**
   - *A:* The 64-dimensional feature space enables full-rank analytical covariance decomposition without numerical truncation, directly linking every concept to the MAT2303 curriculum while avoiding black-box neural networks.

4. **Q: How is data leakage avoided?**
   - *A:* The mean vector $\boldsymbol{\mu}$, covariance matrix $\mathbf{C}$, eigenvector basis $\mathbf{W}_k$, and prototype centroids $\mathbf{p}_c$ are computed strictly on the training partition. Test samples are projected and evaluated using frozen parameters.

---

## 6. Submission Artifacts Verification

- **Jupyter Notebook:** `SamhithVGupta_RollNo_Code.ipynb` (31 cells, 521 KB, executed with outputs & matplotlib charts).
- **CA3 PDF Report:** `SamhithVGupta_RollNo_CA3.pdf` (8 pages, 391 KB, formal cover page, equations, flowchart, scree plot, confusion matrix, metrics table, viva Q&A).
- **Automated Tests:** 18 passing unit & integration tests (`pytest backend/tests`).
