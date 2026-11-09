import React, { useState } from 'react';
import { Download, Expand, ZoomIn, ZoomOut, RotateCcw, ImageOff, ArrowRight } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { Dialog, EmptyState, FacePicker, Notice, ResultNav } from '../components/Shared';
import { downloadFile, verdictName } from '../services/format';

export function ExplainabilityPage() {
  const { analysisResult: result, selectedFace: face, navigateTo } = useAnalysis();
  const [opacity, setOpacity] = useState(65);
  const [zoom, setZoom] = useState(1);
  const [expanded, setExpanded] = useState(null);
  if (!result) return <EmptyState />;
  const original = face?.face_crop_base64;
  const heatmap = face?.gradcam_heatmap_base64;
  const overlay = face?.gradcam_overlay_base64;
  const visuals = [{ key: 'original', label: 'Face crop', image: original }, { key: 'heatmap', label: 'Attention map', image: heatmap }, { key: 'overlay', label: 'Combined view', image: overlay || (original && heatmap ? heatmap : null) }];
  const renderVisual = visual => visual.image ? <div className="evidence-image" style={{ transform: `scale(${zoom})` }}>{visual.key === 'overlay' && original && <img src={original} alt="Original face beneath the attention map" />}<img src={visual.image} alt={visual.label} style={visual.key === 'overlay' && original ? { opacity: opacity / 100 } : undefined} /></div> : <div className="image-unavailable"><ImageOff size={34} /><span>{visual.key === 'original' ? 'No face crop returned' : 'No attention map returned'}</span></div>;
  return <div className="container workspace"><div className="page-heading heading-with-action"><div><h1>See what the model sees.</h1><p>Grad-CAM shows which regions influenced EfficientNet-B0.</p></div><FacePicker /></div><ResultNav />
    {!heatmap && !overlay && <Notice>Visual evidence is unavailable for this face. It may have failed the quality check, or Grad-CAM was switched off for this analysis.</Notice>}
    <div className="evidence-grid">{visuals.map(visual => <figure key={`${face?.face_id}-${visual.key}`}><div className="evidence-frame">{renderVisual(visual)}{visual.image && <button className="icon-button expand-image" onClick={() => setExpanded(visual)} aria-label={`Expand ${visual.label.toLowerCase()}`}><Expand size={18} /></button>}</div><figcaption><strong>{visual.label}</strong>{visual.image && <button className="icon-button" onClick={() => downloadFile(visual.image, `${result.reportId}-face-${face?.face_id || 1}-${visual.key}.jpg`)} aria-label={`Download ${visual.label.toLowerCase()}`}><Download size={18} /></button>}</figcaption></figure>)}</div>
    <div className="evidence-controls"><label htmlFor="overlay-opacity">Overlay strength <output>{opacity}%</output><input id="overlay-opacity" type="range" min="0" max="100" value={opacity} onChange={event => setOpacity(Number(event.target.value))} disabled={!original || (!heatmap && !overlay)} /></label><div className="zoom-controls"><span>Zoom {Math.round(zoom * 100)}%</span><button className="icon-button" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom(value => Math.max(1, value - 0.25))}><ZoomOut size={19} /></button><button className="icon-button" aria-label="Zoom in" disabled={zoom >= 2.5} onClick={() => setZoom(value => Math.min(2.5, value + 0.25))}><ZoomIn size={19} /></button><button className="icon-button" aria-label="Reset zoom" disabled={zoom === 1} onClick={() => setZoom(1)}><RotateCcw size={18} /></button></div></div>
    <div className="result-notes"><section><h2>Attention, explained.</h2><p>Warmer regions contributed more to the model’s prediction{face?.efficientnet?.label ? ` of “${verdictName(face.efficientnet.label)}”` : ''}. Cooler regions contributed less.</p></section><section><h3>Context matters.</h3><p>A highlighted area is not a proven edit. Grad-CAM explains one model’s attention; it does not locate manipulation with pixel-level certainty.</p></section></div>
    <div className="page-next"><p>Keep the images alongside the full analysis.</p><button className="button" onClick={() => navigateTo('report')}>View full report <ArrowRight size={18} /></button></div>
    {expanded && <Dialog title={expanded.label} className="image-dialog" onClose={() => setExpanded(null)}><div className="expanded-evidence">{renderVisual(expanded)}</div><p className="small muted">Face {face?.face_id || 1} · {expanded.key === 'overlay' ? `Overlay strength ${opacity}%` : 'Use the zoom controls on the evidence page to inspect a region.'}</p></Dialog>}
  </div>;
}
