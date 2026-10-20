import React from 'react';
import {
  BrainCircuit,
  Home,
  PenTool,
  BarChart3,
  BookOpen,
  Award,
  FileCode2,
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'analyzer', label: 'Digit Analyzer', icon: PenTool },
    { id: 'dashboard', label: 'Math Dashboard', icon: BarChart3 },
    { id: 'methodology', label: 'Methodology', icon: BookOpen },
    { id: 'evaluation', label: 'Model Evaluation', icon: Award },
    { id: 'docs', label: 'Documentation & Viva', icon: FileCode2 },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-[#070a13]/90 border-b border-slate-800/80 px-4 lg:px-8 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand / Title */}
        <div 
          onClick={() => setActiveTab('home')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 text-white shadow-lg shadow-violet-500/20 group-hover:scale-105 transition-transform">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-violet-950/80 text-violet-300 border border-violet-800/60">
                MAT2303 CA3
              </span>
              <span className="text-xs font-medium text-cyan-400">Linear Algebra & SVD</span>
            </div>
            <h1 className="text-lg font-bold text-white tracking-tight group-hover:text-cyan-300 transition-colors">
              Handwritten Digit Feature Analyzer
            </h1>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="flex items-center gap-1 sm:gap-1.5 overflow-x-auto max-w-full pb-1 md:pb-0 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-violet-600/20 text-cyan-300 border border-violet-500/40 shadow-sm shadow-violet-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
