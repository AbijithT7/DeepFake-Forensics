import React, { useEffect, useRef } from 'react';
import { ArrowRight, Check, FileSearch, X, Info } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { modelNames, percent, verdictName } from '../services/format';

export function Notice({ children, error = false, onClose }) {
  return <div className={`notice ${error ? 'notice-error' : ''}`} role={error ? 'alert' : 'status'}><Info size={19} aria-hidden="true" /><div>{children}</div>{onClose && <button className="icon-button" aria-label="Dismiss message" onClick={onClose}><X size={18} /></button>}</div>;
}
export function EmptyState({ title = 'Start with an image', children = 'Upload a portrait to see its analysis, visual evidence, and report.' }) {
  const { navigateTo, historyReady } = useAnalysis();
  return <div className="empty-state"><FileSearch size={48} strokeWidth={1.2} /><h1>{historyReady ? title : 'Loading your reports…'}</h1><p>{children}</p><button className="button" onClick={() => navigateTo('analyze')}>Analyze an image <ArrowRight size={18} /></button></div>;
}
export function ResultNav() {
  const { activePage } = useAnalysis();
  return <nav className="result-nav" aria-label="Analysis sections">{[['results', 'Overview'], ['explainability', 'Visual evidence'], ['indicators', 'Diagnostics'], ['report', 'Full report']].map(([page, title]) => <a key={page} href={`#/${page}`} aria-current={activePage === page ? 'page' : undefined}>{title}</a>)}</nav>;
}
export function FacePicker() {
  const { analysisResult, selectedFace, setSelectedFaceId } = useAnalysis();
  const faces = analysisResult?.predictions?.filter(face => face.face_id > 0) || [];
  return faces.length > 1 ? <label className="face-picker">Inspect face<select value={selectedFace?.face_id || ''} onChange={event => setSelectedFaceId(Number(event.target.value))}>{faces.map(face => <option key={face.face_id} value={face.face_id}>Face {face.face_id} — {verdictName(face.label)}</option>)}</select></label> : null;
}
export function Verdict({ value, large = false }) {
  return <span className={`${large ? 'verdict-large' : 'verdict'} verdict-${(value || '').toLowerCase()}`}>{verdictName(value)}</span>;
}
export function ModelTable({ prediction }) {
  return <div className="table-scroll"><table className="model-table"><thead><tr><th scope="col">Model</th><th scope="col">Signal</th><th scope="col">Fake score</th><th scope="col">Reference threshold</th><th scope="col">Input</th></tr></thead><tbody>{modelNames.filter(([key]) => key !== 'authenticity' || prediction?.authenticity).map(([key, label]) => {
    const result = prediction?.[key];
    const available = result && result.loaded !== false && result.score != null;
    const fakeScore = result?.probabilities?.FAKE;
    const threshold = prediction?.thresholds?.[key];
    const diagnostic = ['diagnostic', 'classifier_input'].includes(result?.decision_role);
    const classifierVerdict = key === 'authenticity' && result?.decision_role === 'primary_classifier' ? prediction?.label : null;
    return <tr key={key}><th scope="row">{label}{diagnostic && <div className="small muted">{result?.decision_role === 'classifier_input' ? 'Decision input' : 'Diagnostic only'}</div>}</th><td>{available ? classifierVerdict ? <Verdict value={classifierVerdict} /> : diagnostic ? <span className="muted">{threshold != null && fakeScore >= threshold ? 'High fake signal' : 'Below threshold'}</span> : threshold != null ? fakeScore >= threshold ? <Verdict value="FAKE" /> : <span className="muted">Below threshold</span> : <Verdict value={result.label} /> : <span className="muted">Unavailable</span>}</td><td><div className="score-cell"><span>{available ? percent(fakeScore) : '—'}</span>{available && <meter min="0" max="1" value={fakeScore ?? 0} aria-label={`${label} fake score`} />}</div></td><td>{threshold != null ? percent(threshold) : '—'}</td><td>{result?.input_scope === 'image' ? 'Whole image' : available ? result?.face_id ? `Face ${result.face_id}` : 'Face crop' : '—'}</td></tr>;
  })}</tbody></table></div>;
}
export function Dialog({ title, children, onClose, className = '' }) {
  const ref = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    ref.current.showModal();
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = oldOverflow; previous?.focus(); };
  }, []);
  return <dialog ref={ref} className={`dialog ${className}`} onCancel={event => { event.preventDefault(); onClose(); }} onClick={event => { if (event.target === ref.current) onClose(); }} aria-labelledby="dialog-title"><div className="dialog-header"><h2 id="dialog-title">{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X /></button></div>{children}</dialog>;
}
export function CheckItem({ children }) { return <li><Check size={19} aria-hidden="true" />{children}</li>; }
