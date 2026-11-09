import React, { useEffect, useRef } from 'react';
import { AnalysisProvider, useAnalysis } from './context/AnalysisContext';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { Notice } from './components/Shared';
import { LandingPage } from './pages/LandingPage';
import { AnalyzePage } from './pages/AnalyzePage';
import { ResultsPage } from './pages/ResultsPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { IndicatorsPage } from './pages/IndicatorsPage';
import { ReportPage } from './pages/ReportPage';
import { HistoryPage } from './pages/HistoryPage';
import { AboutPage } from './pages/AboutPage';

const pageComponents = { landing: LandingPage, analyze: AnalyzePage, results: ResultsPage, explainability: ExplainabilityPage, indicators: IndicatorsPage, report: ReportPage, history: HistoryPage, about: AboutPage };
const pageTitles = { landing: 'Deepfake forensics', analyze: 'Analyze an image', results: 'Analysis results', explainability: 'Visual evidence', indicators: 'Diagnostics', report: 'Forensic report', history: 'Saved reports', about: 'About' };
function Content() {
  const { activePage, storageWarning, setStorageWarning, isAnalyzing, navigateTo } = useAnalysis();
  const main = useRef(null);
  const firstRender = useRef(true);
  const Page = pageComponents[activePage] || LandingPage;
  useEffect(() => {
    document.title = `DeepTrace — ${pageTitles[activePage]}`;
    if (!firstRender.current) main.current?.focus({ preventScroll: true });
    firstRender.current = false;
    const observer = new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); } }), { threshold: 0.08 });
    main.current?.querySelectorAll('[data-reveal]').forEach(element => observer.observe(element));
    return () => observer.disconnect();
  }, [activePage]);
  return <div className="app-shell"><a className="skip-link" href="#main-content" onClick={event => { event.preventDefault(); main.current?.focus(); }}>Skip to content</a><Header />
    {storageWarning && <div className="container storage-notice"><Notice onClose={() => setStorageWarning(null)}>{storageWarning}</Notice></div>}
    {isAnalyzing && activePage !== 'analyze' && <div className="container"><button className="analysis-banner" onClick={() => navigateTo('analyze')}>Your image is being analyzed. Return to analysis →</button></div>}
    <main id="main-content" ref={main} tabIndex={-1}><Page /></main><Footer /></div>;
}
export function App() { return <AnalysisProvider><Content /></AnalysisProvider>; }
export default App;
