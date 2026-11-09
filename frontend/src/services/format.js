export const modelNames = [
  ['authenticity', 'DINOv2 auxiliary classifier'],
  ['efficientnet', 'EfficientNet-B0'], ['vit', 'Original ViT'], ['community_vit', 'Community Forensics ViT'], ['sbi', 'SBI EfficientNet-B4'], ['hybrid_srm', 'Hybrid SRM'], ['deepguard', 'DeepGuard'],
];
export const percent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Not available';
export const formatBytes = size => !Number.isFinite(size) ? 'Not available' : size < 1024 * 1024 ? `${Math.round(size / 1024)} KB` : `${(size / (1024 * 1024)).toFixed(1)} MB`;
export const formatDate = value => value && !Number.isNaN(Date.parse(value)) ? new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : 'Not recorded';
export const readable = value => value ? String(value).replaceAll('_', ' ').toLowerCase().replace(/^./, char => char.toUpperCase()) : 'Not available';
export const verdictName = value => ({ REAL: 'Real', FAKE: 'Fake', INCONCLUSIVE: 'Inconclusive' }[value] || 'Unavailable');
export function downloadFile(url, name) {
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name;
  document.body.append(anchor); anchor.click(); anchor.remove();
}
export function downloadJson(data, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  downloadFile(url, name);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
