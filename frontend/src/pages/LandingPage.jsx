import React, { useState } from 'react';
import { ArrowRight, ArrowDown, ScanFace, Layers3, ChartNoAxesCombined, FileText, MoveUpRight } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { samples } from '../services/api';
import sculpture from '../assets/sculpture.webp';
import exampleFace from '../assets/example-face.jpg';
import exampleAttention from '../assets/example-attention.jpg';
import exampleOverlay from '../assets/example-overlay.jpg';

export function LandingPage() {
  const { navigateTo, handleLoadSample } = useAnalysis();
  const [sample, setSample] = useState('fake');
  const [detail, setDetail] = useState(false);
  return <>
    <section className="container hero">
      <div className="hero-copy">
        <h1 className="hero-title enter">Look beyond<br />the surface.</h1>
        <p className="hero-description enter">Deepfake forensics, made clear. Examine a face, explore the evidence, and see what the pixels reveal.</p>
        <div className="hero-actions enter"><button className="button" onClick={() => navigateTo('analyze')}>Analyze an image <ArrowRight size={20} /></button><button className="text-link" onClick={() => document.getElementById('try-it').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' })}>Try a sample <ArrowDown size={17} /></button></div>
        <p className="hero-note enter">JPG, PNG, WebP & BMP · Up to 10 MB</p>
      </div>
      <figure className="hero-art enter">
        <img className="sculpture" src={sculpture} alt="Classical marble sculpture, a study of a human face" width="900" height="1000" fetchPriority="high" />
        <div className="art-focus" aria-hidden="true"><span /><span /><span /><span /><div className="scan-line" /></div>
      </figure>
    </section>
    <div className="container capability-strip" aria-label="Analysis capabilities">
      {[[ScanFace, 'Face-first analysis', 'SCRFD localization'], [Layers3, 'Five perspectives', 'Specialist detectors'], [ChartNoAxesCombined, 'Visible evidence', 'Grad-CAM attribution'], [FileText, 'The complete picture', 'Downloadable reports']].map(([Icon, title, description]) => <div className="capability enter" key={title}><Icon size={28} strokeWidth={1.3} aria-hidden="true" /><div><strong>{title}</strong><span>{description}</span></div></div>)}
    </div>
    <section id="try-it" className="sample-section">
      <div className="container sample-layout">
        <div className="sample-copy">
          <h2 data-reveal>Take a closer look.</h2>
          <p data-reveal>Real or manipulated? Start with one of our sample portraits. Follow the same analysis your own images go through.</p>
          <div className="sample-options" data-reveal aria-label="Choose a sample portrait">{[['fake', 'Manipulated portrait', 'A labeled fake sample'], ['real', 'Real portrait', 'A photographed face']].map(([type, title, subtitle]) => <button key={type} className={`sample-option ${sample === type ? 'selected' : ''}`} aria-pressed={sample === type} onClick={() => { setSample(type); setDetail(false); }}><img src={samples[type]} alt={subtitle} width="72" height="72" loading="lazy" /><span><strong>{title}</strong><span>{subtitle}</span></span><ArrowRight size={19} /></button>)}</div>
          <button data-reveal className="button" onClick={() => handleLoadSample(sample)}>Analyze this sample <ArrowRight size={19} /></button>
          <p className="small muted" data-reveal>These are labeled examples, not precomputed results. The analysis runs when you start it.</p>
        </div>
        <div className="sample-viewer" data-ui data-reveal>
          <div className="sample-viewer-top"><span>Explore the portrait</span><ScanFace size={20} /></div>
          <div className={`sample-portrait ${detail ? 'show-detail' : ''}`}><img key={sample} src={samples[sample]} alt={`${sample === 'fake' ? 'Manipulated' : 'Real'} sample portrait ${detail ? 'enlarged for a closer look' : ''}`} width="600" height="600" loading="lazy" /><div className="portrait-reticle" aria-hidden="true" /></div>
          <div className="sample-viewer-bottom"><div className="segmented"><button aria-pressed={!detail} onClick={() => setDetail(false)}>Original</button><button aria-pressed={detail} onClick={() => setDetail(true)}>Closer look</button></div><span>Preview only</span></div>
        </div>
      </div>
    </section>
    <section className="container approach-section">
      <h2 data-reveal>Evidence you can inspect.</h2>
      <div className="approach-body"><p data-reveal>DeepTrace brings several detectors into one place, with face-level results and visual explanations you can inspect yourself.</p><p data-reveal>A model score is a signal, not proof. See where models agree, where they differ, and when the image does not support a clear answer.</p><a data-reveal href="#/about" className="text-link">Understand the process <MoveUpRight size={18} /></a></div>
      <div className="example-evidence">{[[exampleFace, 'The face'], [exampleAttention, 'The model’s attention'], [exampleOverlay, 'The two, together']].map(([src, label]) => <figure data-reveal key={label}><img src={src} alt={`${label}: actual output from the project’s sample analysis`} width="350" height="350" loading="lazy" /><figcaption>{label}</figcaption></figure>)}</div>
      <p data-reveal className="example-caption small muted">An actual sample run through EfficientNet-B0. The heatmap shows model attention, not a confirmed edit.</p>
    </section>
  </>;
}
