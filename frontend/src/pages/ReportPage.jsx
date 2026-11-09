import React, { useRef, useState } from 'react';
import { Download, FileJson, LoaderCircle, Scan, Printer } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { EmptyState, ModelTable, Notice, ResultNav, Verdict } from '../components/Shared';
import { downloadJson, formatBytes, formatDate, percent, readable } from '../services/format';

export function ReportPage() {
  const { analysisResult: result } = useAnalysis();
  const report = useRef(null);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState(null);
  if (!result) return <EmptyState />;
  const final = result.final_prediction;
  const faces = result.predictions || [];
  async function exportPdf() {
    if (exporting) return;
    setExporting(true); setError(null);
    try {
      const { exportReportPdf } = await import('../services/pdf');
      await exportReportPdf(result);
    } catch {
      setError('The PDF could not be generated. You can use Print / Save as PDF or download the complete JSON report.');
    } finally { setExporting(false); }
  }
  return <div className="container workspace report-workspace"><div className="page-heading heading-with-action no-print"><div><h1>The complete picture.</h1><p>Your image, the evidence, and every model’s perspective.</p></div><button className="button" onClick={exportPdf} disabled={exporting}>{exporting ? <LoaderCircle size={18} className="spin" /> : <Download size={18} />}{exporting ? 'Preparing PDF…' : 'Download PDF'}</button></div><div className="no-print"><ResultNav />{error && <Notice error>{error}</Notice>}</div>
    <article ref={report} className="report-paper">
      <header className="report-header"><div><div className="wordmark"><Scan /><span>DeepTrace.</span></div><h2>Forensic analysis report</h2></div><div><strong>{result.reportId}</strong><span>{formatDate(result.timestamp)}</span></div></header>
      <section className="report-summary">{result.originalImageBase64 && <img src={result.originalImageBase64} alt="Original analyzed image" />}<div><span>Image verdict</span><Verdict value={final.label} large /><p>Model score: <strong>{percent(final.score)}</strong></p>{final.reason && <p>{readable(final.reason)}</p>}</div></section>
      <section className="report-section"><h3>Image details</h3><dl className="report-details"><div><dt>Filename</dt><dd>{result.imageMeta?.name || 'Not recorded'}</dd></div><div><dt>Dimensions</dt><dd>{result.imageMeta?.width ? `${result.imageMeta.width} × ${result.imageMeta.height} px` : 'Not recorded'}</dd></div><div><dt>File size</dt><dd>{formatBytes(result.imageMeta?.size)}</dd></div><div><dt>Faces detected</dt><dd>{result.faces_detected ?? 'Not available'}</dd></div><div><dt>Processing time</dt><dd>{typeof result.processingTimeSeconds === 'number' ? `${result.processingTimeSeconds.toFixed(2)} seconds` : 'Not recorded'}</dd></div><div><dt>Threshold</dt><dd>{percent(result.uncertainty_threshold)}</dd></div></dl></section>
      <section className="report-section"><h3>Image detection signals</h3><ModelTable prediction={final} />{final.triggering_models?.length > 0 && <p>Flagged by: {final.triggering_models.join(", ")}.</p>}</section>{faces.map((face, index) => <section className="report-section report-face" key={`${face.face_id}-${index}`}><div className="report-face-header"><h3>{face.face_id > 0 ? `Face ${face.face_id}` : 'Image assessment'}</h3><Verdict value={face.label} /></div>{face.reason && <p>{readable(face.reason)}</p>}<ModelTable prediction={face} />{(face.face_crop_base64 || face.gradcam_heatmap_base64 || face.gradcam_overlay_base64) && <div className="report-images">{[['Face crop', face.face_crop_base64], ['Grad-CAM attention', face.gradcam_heatmap_base64], ['Combined view', face.gradcam_overlay_base64]].filter(([, image]) => image).map(([label, image]) => <figure key={label}><img src={image} alt={label} /><figcaption>{label}</figcaption></figure>)}</div>}{face.fusion?.policy && <p><strong>Decision policy:</strong> {face.fusion.policy}</p>}{face.box && <p><strong>Face bounds:</strong> {face.box.join(', ')} (x, y, width, height).</p>}{face.combined_result?.votes && <p><strong>Supplementary specialist votes:</strong> {face.combined_result.votes.FAKE || 0} fake, {face.combined_result.votes.REAL || 0} real. {readable(face.combined_result.consensus)}.</p>}{face.shadow_routing && <div className="report-routing"><h4>Routing diagnostics</h4><p>Image type: {readable(face.shadow_routing.image_type)} · Router score: {percent(face.shadow_routing.router_confidence)}.</p><p>Mode: {face.shadow_routing.shadow_mode ? 'Shadow (does not change the image verdict)' : 'Production'}. Policy: {readable(face.shadow_routing.routing_policy)}.</p><p>Selected specialists: {face.shadow_routing.selected_specialists?.join(', ') || 'Not available'}.</p>{face.shadow_routing.policy_reason && <p>{face.shadow_routing.policy_reason}</p>}</div>}</section>)}
      <section className="report-section report-limitations"><h3>Observations & limitations</h3><p>{result.message || 'Analysis completed using the available models.'}</p><p>Image detectors analyze the whole photograph. Face specialists analyze usable crops. A qualified fake signal flags the image; missing or contrary models cannot veto it.</p><p>Scores are uncalibrated model outputs, not verified probabilities of authenticity. Grad-CAM indicates model attention, not a confirmed manipulation boundary. Compression, image quality, and unfamiliar generation methods can affect results. This report is supporting evidence and should be assessed alongside the image’s source and context.</p></section>
      <footer className="report-footer"><span>DeepTrace · Image forensics</span><span>{result.reportId}</span></footer>
    </article>
    <div className="report-downloads no-print"><p>Need the underlying data?</p><div className="button-row"><button className="button button-secondary" onClick={() => window.print()}><Printer size={18} />Print / Save as PDF</button><button className="button button-secondary" onClick={() => downloadJson(result, `${result.reportId}.json`)}><FileJson size={18} />Download JSON</button></div></div>
  </div>;
}
