import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { checkHealth, predictImage, fetchSampleImage } from '../services/api';
import { readReports, saveReport, removeReport, removeAllReports } from '../services/storage';

const AnalysisContext = createContext(null);
const pages = new Set(['landing', 'analyze', 'results', 'explainability', 'indicators', 'report', 'history', 'about']);
const readPage = () => { const page = location.hash.replace(/^#\/?/, ''); return pages.has(page) ? page : 'landing'; };

export function AnalysisProvider({ children }) {
  const [activePage, setActivePage] = useState(readPage);
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'dark');
  const [currentFile, setCurrentFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageMeta, setImageMeta] = useState({});
  const [isSelecting, setIsSelecting] = useState(false);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState(0.6);
  const [includeGradcam, setIncludeGradcam] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [selectedFaceId, setSelectedFaceId] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyReady, setHistoryReady] = useState(false);
  const [storageWarning, setStorageWarning] = useState(null);
  const [systemHealth, setSystemHealth] = useState({ status: 'checking' });
  const request = useRef(null);
  const selection = useRef(0);
  const startedAt = useRef(0);

  const navigateTo = useCallback(page => {
    const next = pages.has(page) ? page : 'landing';
    if (readPage() !== next) location.hash = next === 'landing' ? '/' : `/${next}`;
    setActivePage(next);
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, []);
  useEffect(() => {
    const onHash = () => { setActivePage(readPage()); window.scrollTo({ top: 0, behavior: 'instant' }); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#10120f' : '#f6f7f2');
    try { localStorage.setItem('deeptrace-theme', theme); } catch { /* Theme still works without storage. */ }
  }, [theme]);
  const refreshHealth = useCallback(async () => setSystemHealth(await checkHealth()), []);
  useEffect(() => {
    let alive = true;
    const refresh = async () => { const data = await checkHealth(); if (alive) setSystemHealth(data); };
    refresh();
    const timer = setInterval(refresh, 30000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  useEffect(() => {
    let alive = true;
    readReports().then(items => {
      if (!alive) return;
      setHistory(items);
      setAnalysisResult(current => current || items[0]?.fullResult || null);
    }).catch(() => {
      if (alive) setStorageWarning('Browser storage is unavailable. You can analyze images and download reports, but this session will not be saved.');
    }).finally(() => { if (alive) setHistoryReady(true); });
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    if (!isAnalyzing) return;
    const timer = setInterval(() => setElapsed((Date.now() - startedAt.current) / 1000), 250);
    return () => clearInterval(timer);
  }, [isAnalyzing]);
  useEffect(() => () => request.current?.abort(), []);

  async function handleSelectFile(file) {
    if (!file || request.current) return false;
    const token = ++selection.current;
    setAnalysisError(null); setCurrentFile(null); setPreviewUrl(null); setImageMeta({}); setIsSelecting(true);
    try {
      if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) throw new Error('Choose a JPG, PNG, WebP, or BMP image.');
      if (!file.size) throw new Error('This file is empty. Choose another image.');
      if (file.size > 10 * 1024 * 1024) throw new Error('This image is larger than 10 MB. Choose a smaller file.');
      const url = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('The image could not be read. Please choose it again.'));
        reader.readAsDataURL(file);
      });
      const image = new Image(); image.src = url;
      await image.decode().catch(() => { throw new Error('This file is not a readable image. Choose a different JPG, PNG, WebP, or BMP.'); });
      if (image.naturalWidth * image.naturalHeight > 40_000_000) throw new Error('Resize this image below 40 megapixels before uploading.');
      if (token !== selection.current) return false;
      setCurrentFile(file); setPreviewUrl(url);
      setImageMeta({ name: file.name, size: file.size, type: file.type, width: image.naturalWidth, height: image.naturalHeight });
      return true;
    } catch (error) { if (token === selection.current) setAnalysisError(error.message); return false; }
    finally { if (token === selection.current) setIsSelecting(false); }
  }
  async function handleLoadSample(type = 'fake') {
    if (request.current) { navigateTo('analyze'); return; }
    navigateTo('analyze'); setAnalysisError(null); setIsSelecting(true);
    try { await handleSelectFile(await fetchSampleImage(type)); }
    catch (error) { setAnalysisError(error.message); setIsSelecting(false); }
  }
  async function executeAnalysis() {
    if (request.current || isSelecting) return;
    if (!currentFile) { setAnalysisError('Choose an image before starting an analysis.'); return; }
    const controller = new AbortController(); request.current = controller;
    setIsAnalyzing(true); setAnalysisError(null); setElapsed(0); startedAt.current = Date.now();
    const timeout = setTimeout(() => controller.abort('timeout'), 180000);
    try {
      const result = await predictImage(currentFile, uncertaintyThreshold, includeGradcam, controller.signal);
      if (controller.signal.aborted) return;
      const suffix = (globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)).slice(0, 4).toUpperCase();
      const reportId = `DT-${Date.now().toString(36).toUpperCase()}-${suffix}`;
      const enriched = { ...result, reportId, timestamp: new Date().toISOString(), processingTimeSeconds: (Date.now() - startedAt.current) / 1000, imageMeta, originalImageBase64: previewUrl };
      const item = { id: reportId, timestamp: enriched.timestamp, filename: currentFile.name, verdict: result.final_prediction.label, score: result.final_prediction.score, facesDetected: result.faces_detected, thumbnail: result.final_prediction.face_crop_base64 || previewUrl, fullResult: enriched };
      setAnalysisResult(enriched); setSelectedFaceId(null); setHistory(items => [item, ...items]);
      try { await saveReport(item); } catch { setStorageWarning('This report could not be saved in your browser. Download a copy to keep it.'); }
      navigateTo('results');
    } catch (error) {
      if (controller.signal.aborted) {
        if (controller.signal.reason === 'timeout') setAnalysisError('The service took longer than three minutes. Please try again.');
      } else { setAnalysisError(error.message); refreshHealth(); }
    } finally {
      clearTimeout(timeout);
      if (request.current === controller) { request.current = null; setIsAnalyzing(false); }
    }
  }
  function cancelAnalysis() {
    request.current?.abort(); request.current = null; setIsAnalyzing(false);
    setAnalysisError('Stopped waiting for this analysis. The server may finish processing the request. You can try again.');
  }
  function clearFile() {
    if (request.current) return;
    ++selection.current;
    setCurrentFile(null); setPreviewUrl(null); setImageMeta({}); setAnalysisError(null); setIsSelecting(false);
  }
  function resetAnalysis() { if (!request.current) clearFile(); navigateTo('analyze'); }
  function loadHistoryItem(item) { setAnalysisResult(item.fullResult); setSelectedFaceId(null); navigateTo('results'); }
  async function deleteHistoryItem(id) {
    try { await removeReport(id); setHistory(items => items.filter(item => item.id !== id)); }
    catch { setStorageWarning('The report could not be removed. Please try again.'); }
  }
  async function clearHistory() {
    try { await removeAllReports(); setHistory([]); setAnalysisResult(null); }
    catch { setStorageWarning('Reports could not be cleared. Please try again.'); }
  }
  const selectedFace = analysisResult?.predictions?.find(face => face.face_id === selectedFaceId) || analysisResult?.predictions?.find(face => face.face_id > 0) || analysisResult?.final_prediction;
  return <AnalysisContext.Provider value={{ activePage, navigateTo, theme, toggleTheme: () => setTheme(value => value === 'dark' ? 'light' : 'dark'), currentFile, previewUrl, imageMeta, handleSelectFile, handleLoadSample, clearFile, executeAnalysis, cancelAnalysis, resetAnalysis, isAnalyzing, isSelecting, elapsed, analysisResult, analysisError, setAnalysisError, uncertaintyThreshold, setUncertaintyThreshold, includeGradcam, setIncludeGradcam, history, historyReady, loadHistoryItem, deleteHistoryItem, clearHistory, storageWarning, setStorageWarning, systemHealth, refreshHealth, selectedFace, selectedFaceId, setSelectedFaceId }}>{children}</AnalysisContext.Provider>;
}
export const useAnalysis = () => useContext(AnalysisContext);
