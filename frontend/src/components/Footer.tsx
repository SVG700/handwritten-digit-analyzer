import React from 'react';
import { Layers, ShieldCheck, Cpu } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-[#070a13] py-8 px-4 lg:px-8 mt-16 text-slate-400 text-sm">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-white">MAT2303 CA3 Project</span>
            <span className="text-slate-600">•</span>
            <span className="text-cyan-400">Handwritten Digit Feature & Similarity Analyzer</span>
          </div>
          <p className="text-xs text-slate-500">
            Student: <span className="text-slate-300 font-medium">Samhith V Gupta</span> | Roll No: <span className="text-violet-400 font-mono">[ROLL_NO_PLACEHOLDER]</span>
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300">
            <Layers className="w-3.5 h-3.5 text-violet-400" />
            <span>PCA / SVD Projection</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>No Data Leakage</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Real 8×8 Matrix Math</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
