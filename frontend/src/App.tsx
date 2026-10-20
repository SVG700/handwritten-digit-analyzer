import { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { HomePage } from './pages/HomePage';
import { AnalyzerPage } from './pages/AnalyzerPage';
import { DashboardPage } from './pages/DashboardPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { DocsPage } from './pages/DocsPage';
import type { MathPipelineResponse } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [latestResult, setLatestResult] = useState<MathPipelineResponse | null>(null);

  return (
    <div className="min-h-screen bg-[#070a13] text-slate-100 flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'home' && <HomePage setActiveTab={setActiveTab} />}
        {activeTab === 'analyzer' && (
          <AnalyzerPage
            setActiveTab={setActiveTab}
            latestResult={latestResult}
            setLatestResult={setLatestResult}
          />
        )}
        {activeTab === 'dashboard' && (
          <DashboardPage
            latestResult={latestResult}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === 'methodology' && <MethodologyPage />}
        {activeTab === 'evaluation' && <EvaluationPage />}
        {activeTab === 'docs' && <DocsPage />}
      </main>

      <Footer />
    </div>
  );
}

export default App;
