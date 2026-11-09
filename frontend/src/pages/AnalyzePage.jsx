import React, { useRef, useState } from 'react';
import { Upload, ArrowRight, ImagePlus, X, LoaderCircle, ScanFace, Check, SlidersHorizontal, RefreshCw } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { Notice } from '../components/Shared';
import { samples } from '../services/api';
import { formatBytes } from '../services/format';

export function AnalyzePage() {
  const { currentFile, previewUrl, imageMeta, handleSelectFile, handleLoadSample, clearFile, executeAnalysis, cancelAnalysis, isAnalyzing, isSelecting, elapsed, analysisError, setAnalysisError, uncertaintyThreshold, setUncertaintyThreshold, includeGradcam, setIncludeGradcam, systemHealth, refreshHealth } = useAnalysis();
  const input = useRef(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const offline = systemHealth.status === 'offline';
  const pick = files => {
    if (files.length > 1) { setAnalysisError('Choose one image at a time.'); return; }
    if (files[0]) handleSelectFile(files[0]);
  };
  return <div className="container workspace upload-workspace">
    <div className="page-heading"><h1>{isAnalyzing ? 'Looking a little deeper.' : 'Let’s look closer.'}</h1><p>{isAnalyzing ? 'Your image is being checked for signs of manipulation.' : 'Upload a face image. We’ll help you examine the evidence.'}</p></div>
    {isAnalyzing ? <section className="processing-layout" aria-busy="true">
      <div className="processing-image"><img src={previewUrl} alt="Image currently being analyzed" /><div className="processing-scan" aria-hidden="true" /><ScanFace className="processing-mark" size={36} /></div>
      <div className="processing-copy"><div className="processing-status" role="status"><LoaderCircle className="spin" size={24} /><strong>Analysis in progress</strong></div><p>The service detects faces, runs the available specialists, and prepares your results{includeGradcam ? ' and visual evidence' : ''}.</p><ul className="processing-list"><li><Check size={19} />Image validated</li><li><LoaderCircle size={19} className="spin" />Waiting for the analysis service</li><li className="muted">Results appear when processing finishes</li></ul><div className="indeterminate-bar" aria-hidden="true"><span /></div><div className="processing-meta"><span>{elapsed.toFixed(1)} seconds elapsed</span><span>{imageMeta.name}</span></div><p className="small muted">Large images and CPU processing can take longer.</p><button className="button button-secondary" onClick={cancelAnalysis}>Cancel analysis <X size={17} /></button></div>
    </section> : <div className="upload-layout">
      <div className="upload-main">
        <input ref={input} id="image-upload" className="visually-hidden" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp" aria-label="Choose image to analyze" disabled={isSelecting} onChange={event => { pick(event.target.files); event.target.value = ''; }} />
        <div className={`dropzone ${dragging ? 'dragging' : ''} ${currentFile ? 'has-image' : ''}`} onDragEnter={event => { event.preventDefault(); dragDepth.current++; setDragging(true); }} onDragOver={event => event.preventDefault()} onDragLeave={event => { event.preventDefault(); if (--dragDepth.current <= 0) setDragging(false); }} onDrop={event => { event.preventDefault(); dragDepth.current = 0; setDragging(false); pick(event.dataTransfer.files); }}>
          {currentFile ? <><img className="upload-preview" src={previewUrl} alt={`Preview of ${imageMeta.name}`} /><button className="icon-button remove-image" onClick={clearFile} aria-label="Remove selected image"><X /></button><div className="image-ready"><Check size={17} />Ready to analyze</div></> : <button className="dropzone-target" onClick={() => input.current.click()} disabled={isSelecting}>{isSelecting ? <LoaderCircle className="spin" size={46} /> : <div className="upload-illustration" aria-hidden="true"><ImagePlus size={62} strokeWidth={1.2} /><span><Upload size={22} /></span></div>}<strong>{isSelecting ? 'Preparing your image…' : dragging ? 'Drop your image here' : 'Drop an image here'}</strong><span>{isSelecting ? 'Checking the file and loading a preview.' : <>or <span className="accent-text">browse files</span> from your device</>}</span><small>JPG, PNG, WebP or BMP · Max. 10 MB</small></button>}
        </div>
        {currentFile && <div className="file-details"><div><strong>{imageMeta.name}</strong><span>{imageMeta.width} × {imageMeta.height} pixels · {formatBytes(imageMeta.size)}</span></div><button className="text-link" onClick={() => input.current.click()}>Change image</button></div>}
        {analysisError && <Notice error onClose={() => setAnalysisError(null)}>{analysisError}</Notice>}
        <button className="options-toggle" aria-expanded={optionsOpen} onClick={() => setOptionsOpen(value => !value)}><SlidersHorizontal size={18} /><span>Analysis options</span><span>{optionsOpen ? '−' : '+'}</span></button>
        {optionsOpen && <div className="analysis-options"><label htmlFor="threshold">Minimum fake score <output>{Math.round(uncertaintyThreshold * 100)}%</output></label><input id="threshold" type="range" min="0.5" max="1" step="0.05" value={uncertaintyThreshold} onChange={event => setUncertaintyThreshold(Number(event.target.value))} /><p className="small muted">The decision model uses the higher of this value and its saved fake threshold. Raising it reduces sensitivity. The real threshold stays fixed.</p><label className="checkbox-label"><input type="checkbox" checked={includeGradcam} onChange={event => setIncludeGradcam(event.target.checked)} /><span>Include Grad-CAM visual evidence</span></label></div>}
        <button className="button analyze-submit" onClick={executeAnalysis} disabled={!currentFile || isSelecting}>Analyze image <ArrowRight size={20} /></button>
        <p className="upload-privacy small muted">Your image is sent to the analysis server. Saved reports stay in this browser until you remove them.</p>
      </div>
      <aside className="upload-aside">
        <h2>No image in mind?</h2><p>Get a feel for the process with a sample.</p>
        <div className="upload-samples">{[['fake', 'Manipulated portrait'], ['real', 'Real portrait']].map(([type, label]) => <button key={type} className="upload-sample" onClick={() => handleLoadSample(type)} disabled={isSelecting}><img src={samples[type]} alt={label} width="100" height="100" /><span>{label}</span><ArrowRight size={17} /></button>)}</div>
        <div className="good-image"><h3>A good image helps.</h3><p>Use a clear, front-facing portrait. Keep the face visible and avoid heavy filters or very small images.</p><p>Multiple faces? Each detected face gets its own analysis.</p></div>
        <div className={`service-state ${offline ? 'is-offline' : ''}`}><span className="status-dot" /><span>{systemHealth.status === 'checking' ? 'Connecting to analysis service' : offline ? 'Analysis service is offline' : systemHealth.status === 'degraded' ? 'Some models are unavailable' : 'Analysis service is ready'}</span>{offline && <button className="icon-button" aria-label="Retry service connection" disabled={retrying} onClick={async () => { setRetrying(true); await refreshHealth(); setRetrying(false); }}><RefreshCw size={17} className={retrying ? 'spin' : ''} /></button>}</div>
      </aside>
    </div>}
  </div>;
}
