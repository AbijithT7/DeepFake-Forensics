import React from 'react';
import { useAnalysis } from '../context/AnalysisContext';
import { EmptyState, FacePicker, ModelTable, ResultNav, Verdict } from '../components/Shared';
import { percent, readable } from '../services/format';

export function IndicatorsPage() {
  const { analysisResult: result, selectedFace: face } = useAnalysis();
  if (!result) return <EmptyState />;
  const router = face?.shadow_routing || result.shadow_routing;
  return <div className="container workspace"><div className="page-heading heading-with-action"><div><h1>A little more detail.</h1><p>Inspect the measurements and routing information returned by the service.</p></div><FacePicker /></div><ResultNav />
    <div className="diagnostic-grid"><section className="surface-panel"><h2>Face & image</h2><dl className="detail-list"><div><dt>Detected faces</dt><dd>{result.faces_detected ?? 'Not available'}</dd></div><div><dt>Selected face</dt><dd>{face?.face_id > 0 ? `Face ${face.face_id}` : 'None detected'}</dd></div><div><dt>Face bounds (x, y, w, h)</dt><dd>{face?.box?.join(', ') || 'Not available'}</dd></div><div><dt>Quality result</dt><dd>{face?.reason ? readable(face.reason) : face?.face_id > 0 ? 'Face evaluated' : 'Not available'}</dd></div><div><dt>Confidence threshold</dt><dd>{percent(result.uncertainty_threshold)}</dd></div><div><dt>Compute device</dt><dd>{result.execution_device || 'Not available'}</dd></div></dl></section>
    <section className="surface-panel"><h2>Image-type routing</h2>{router ? <><dl className="detail-list"><div><dt>Estimated type</dt><dd>{readable(router.image_type)}</dd></div><div><dt>Router score</dt><dd>{percent(router.router_confidence)}</dd></div><div><dt>Operating mode</dt><dd>{router.shadow_mode ? 'Shadow (diagnostic only)' : 'Production'}</dd></div><div><dt>Routing policy</dt><dd>{readable(router.routing_policy)}</dd></div><div><dt>Estimate accepted</dt><dd>{router.is_rejected ? 'No' : 'Yes'}</dd></div></dl><p className="small muted">{router.policy_reason || router.rejection_reason || (router.shadow_mode ? 'Shadow routing records an alternative decision without changing the image verdict.' : 'Routing metadata reflects the backend’s active policy.')}</p></> : <p>The service did not return image-type routing information.</p>}</section></div>
    {router && <section className="routing-detail"><h2>Behind the routing choice</h2><p>Selected specialists: {router.selected_specialists?.join(', ') || 'Not available'}.</p>{router.shadow_verdict && <p className="inline-verdict">Hypothetical routed verdict <Verdict value={router.shadow_verdict.label} /></p>}{router.router_probabilities && <div className="probability-list">{Object.entries(router.router_probabilities).sort(([, a], [, b]) => b - a).map(([label, value]) => <div key={label}><span>{readable(label)}</span><meter value={value} min="0" max="1" aria-label={`${readable(label)} router score`} /><strong>{percent(value)}</strong></div>)}</div>}</section>}
    <div className="section-heading"><h2>Specialist outputs</h2></div><ModelTable prediction={face} />
  </div>;
}
