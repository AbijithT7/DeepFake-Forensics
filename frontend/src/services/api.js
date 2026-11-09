import realSample from '../assets/real-sample.jpg';
import fakeSample from '../assets/fake-sample.png';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
export const samples = { real: realSample, fake: fakeSample };

export async function checkHealth(signal) {
  try {
    const response = await fetch(`${API_BASE}/health`, { signal: signal || AbortSignal.timeout(8000) });
    if (!response.ok) throw new Error('The analysis service is unavailable.');
    const data = await response.json();
    if (!['healthy', 'degraded'].includes(data.status)) throw new Error('Unexpected health response.');
    return data;
  } catch { return { status: 'offline' }; }
}

export async function predictImage(file, threshold, includeGradcam, signal) {
  const body = new FormData();
  body.append('file', file);
  let response;
  try {
    response = await fetch(`${API_BASE}/predict/image?uncertainty_threshold=${threshold}&include_gradcam=${includeGradcam}`, { method: 'POST', body, signal });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Could not reach the analysis service. Check your connection and make sure the backend is running, then try again.');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail : Array.isArray(data?.detail) ? data.detail.map(item => item.msg).join('. ') : null;
    throw new Error(detail || (response.status === 503 ? 'The models are still loading. Please try again shortly.' : `The analysis could not finish (${response.status}). Please try again.`));
  }
  if (!data?.final_prediction || !Array.isArray(data.predictions)) throw new Error('The service returned an incomplete result. Please try again.');
  return data;
}

export async function fetchSampleImage(type) {
  const sampleType = type === 'real' ? 'real' : 'fake';
  const response = await fetch(samples[sampleType]);
  if (!response.ok) throw new Error('The sample could not be loaded. Try uploading an image instead.');
  const metadata = sampleType === 'real'
    ? { name: 'trump.jpg', type: 'image/jpeg' }
    : { name: 'fake_face_sample.png', type: 'image/png' };
  return new File([await response.blob()], metadata.name, { type: metadata.type });
}
