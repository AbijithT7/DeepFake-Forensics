import React from 'react';
import { ArrowRight, Plus, ScanFace, FileText } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { EmptyState, FacePicker, ModelTable, ResultNav, Verdict, Notice } from '../components/Shared';
import { percent, readable } from '../services/format';

export function ResultsPage() {
  const { analysisResult: result, selectedFace: face, resetAnalysis, navigateTo, systemHealth } = useAnalysis();
  if (!result) return <EmptyState />;
  const final = result.final_prediction;
  const combined = face?.combined_result;
  const votes = combined?.votes;
  const router = face?.shadow_routing || result.shadow_routing;
  const trained = final.verdict_engine === 'v5';
  const verdictExplanation = trained
    ? { FAKE: 'The detectors provide corroborating evidence of a generated or manipulated image.', REAL: 'Several detectors support a real result for this image.', INCONCLUSIVE: 'The available signals do not provide sufficient corroboration.' }[final.label]
    : final.label === 'FAKE' ? 'A forensic specialist found a fake signal above its threshold.' : final.label === 'REAL' ? 'The primary detectors agree this image appears real.' : 'The available evidence does not support a clear verdict.';
  return <div className="container workspace">
    <div className="page-heading heading-with-action"><div><h1>The results are in.</h1><p className="filename">{result.imageMeta?.name || 'Image analysis'}</p></div><button className="button button-secondary" onClick={resetAnalysis}><Plus size={18} />New analysis</button></div>
    <ResultNav />
    {systemHealth.active_verdict_engine && final.verdict_engine && systemHealth.active_verdict_engine !== final.verdict_engine && <Notice>This saved report used an earlier detector. Run a new analysis to use the current model.</Notice>}
    <section className="verdict-panel"><div className="result-portrait">{result.originalImageBase64 || final.face_crop_base64 ? <img src={result.originalImageBase64 || final.face_crop_base64} alt="Analyzed image" /> : <ScanFace size={64} />}</div><div className="verdict-content"><span className="muted">Image verdict</span><Verdict value={final.label} large /><p>{verdictExplanation}</p><div className="verdict-metrics"><div><span>Model score</span><strong>{percent(final.score)}</strong></div><div><span>Faces detected</span><strong>{result.faces_detected ?? '—'}</strong></div><div><span>Analysis time</span><strong>{typeof result.processingTimeSeconds === 'number' ? `${result.processingTimeSeconds.toFixed(2)} s` : 'Not recorded'}</strong></div></div></div></section>
    {final.reason && <Notice>{readable(final.reason)}. {result.message || 'Try another clear image if more evidence is needed.'}</Notice>}
    <div className="section-heading"><div><h2>Every perspective counts.</h2><p>{trained ? 'The decision rule checks corroboration across detector signals.' : 'Whole-image detector scores and the strongest signal from each face specialist.'}</p></div><FacePicker /></div>
    <ModelTable prediction={final} />
    <div className="result-notes"><section><h3>How to read this result</h3><p>{final.fusion?.policy || 'The image verdict comes from the backend’s primary detector agreement and face quality checks.'}</p><p className="muted">Scores are uncalibrated model outputs, not a probability that an image is authentic. Check the source and context alongside these signals.</p></section><section><h3>Detection signals</h3>{votes ? <><div className="vote-display"><span><b>{votes.FAKE || 0}</b> fake</span><span><b>{votes.REAL || 0}</b> real</span></div><p>{readable(combined.consensus)}. Specialist votes are supplementary; they do not replace the image verdict.</p></> : <p>{final.triggering_models?.length ? `Flagged by: ${final.triggering_models.map(name => name.replaceAll("_", " ")).join(", ")}.` : trained ? final.label === "REAL" ? "The detectors support a real result. Individual model scores can disagree." : "The detectors have insufficient corroboration for a conclusive result." : "No specialist crossed its fake threshold. An inconclusive result does not establish authenticity."}</p>}{router?.image_type && <p className="muted">Image-type estimate: {readable(router.image_type)} ({percent(router.router_confidence)}).</p>}</section></div>
    <div className="page-next"><div><FileText size={28} /><span>Follow the evidence, or keep the full report.</span></div><div className="button-row"><button className="button button-secondary" onClick={() => navigateTo('report')}>View report</button><button className="button" onClick={() => navigateTo('explainability')}>Explore visual evidence <ArrowRight size={18} /></button></div></div>
  </div>;
}
