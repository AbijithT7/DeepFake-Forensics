import { jsPDF } from 'jspdf';
import { formatBytes, formatDate, modelNames, percent, readable, verdictName } from './format';

// Use native PDF text and measured pages: browser screenshots can clip CSS grids
// and split evidence images when a report spans more than one page.
export async function exportReportPdf(result) {
  const pdf = new jsPDF({ unit: 'mm', format: 'a4', compress: true });
  const margin = 17;
  const width = 176;
  const bottom = 273;
  const ink = '#20271b';
  const muted = '#58624e';
  let y = 20;
  const clean = value => String(value ?? 'Not available').replaceAll('×', 'x').replaceAll('’', "'").replace(/[–—]/g, '-').replaceAll('·', ' / ');
  function line(text, x, baseline, size = 10, color = ink, weight = 'normal', options) {
    pdf.setFont('helvetica', weight); pdf.setFontSize(size); pdf.setTextColor(color);
    pdf.text(clean(text), x, baseline, options);
  }
  function newPage() { pdf.addPage(); y = 20; }
  function room(height) { if (y + height > bottom) newPage(); }
  function paragraph(text, { size = 10, color = muted, gap = 4 } = {}) {
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(size);
    const lines = pdf.splitTextToSize(clean(text), width);
    for (const value of lines) { room(5); line(value, margin, y, size, color); y += 5; }
    y += gap;
  }
  function heading(text) { room(16); line(text, margin, y, 15, ink, 'bold'); y += 10; }
  function verdict(label, x, baseline, size = 25) {
    line(verdictName(label), x, baseline, size, label === 'FAKE' ? '#b5332c' : label === 'REAL' ? '#436817' : '#856016', 'bold');
  }
  function addImage(data, x, top, maxWidth, maxHeight) {
    if (!data) return;
    try {
      const properties = pdf.getImageProperties(data);
      const scale = Math.min(maxWidth / properties.width, maxHeight / properties.height);
      const w = properties.width * scale, h = properties.height * scale;
      pdf.addImage(data, properties.fileType, x + (maxWidth - w) / 2, top + (maxHeight - h) / 2, w, h, undefined, 'FAST');
    } catch { line('Image unavailable', x + 3, top + maxHeight / 2, 9, muted); }
  }
  function rows(items) {
    for (const [key, value] of items) {
      pdf.setFontSize(10);
      const values = pdf.splitTextToSize(clean(value), 112);
      room(Math.max(7, values.length * 5));
      line(key, margin, y, 9, muted);
      for (let index = 0; index < values.length; index++) line(values[index], margin + 61, y + index * 5, 10, ink);
      y += Math.max(8, values.length * 5 + 3);
    }
    y += 5;
  }

  line('DeepTrace.', margin, y, 24, ink, 'bold'); y += 13;
  heading('Forensic analysis report');
  paragraph(`${result.reportId} / ${formatDate(result.timestamp)}`, { size: 9 });
  pdf.setFillColor('#f0f3e9'); pdf.rect(margin, y, width, 58, 'F');
  addImage(result.originalImageBase64 || result.final_prediction.face_crop_base64, margin + 5, y + 5, 46, 48);
  line('Image verdict', margin + 59, y + 13, 10, muted);
  verdict(result.final_prediction.label, margin + 59, y + 28);
  line(`Model score: ${percent(result.final_prediction.score)}`, margin + 59, y + 39, 10);
  line(`${result.faces_detected ?? 0} face(s) detected`, margin + 59, y + 47, 10, muted);
  y += 72;
  heading('Image details');
  rows([
    ['Filename', result.imageMeta?.name || 'Not recorded'],
    ['Dimensions', result.imageMeta?.width ? `${result.imageMeta.width} x ${result.imageMeta.height} pixels` : 'Not recorded'],
    ['File size', formatBytes(result.imageMeta?.size)],
    ['Analysis time', typeof result.processingTimeSeconds === 'number' ? `${result.processingTimeSeconds.toFixed(2)} seconds` : 'Not recorded'],
    ['Confidence threshold', percent(result.uncertainty_threshold)],
    ['Compute device', result.execution_device || 'Not recorded'],
  ]);
  heading('Reading this result');
  if (result.final_prediction.reason) paragraph(readable(result.final_prediction.reason));
  paragraph(result.message || 'Analysis completed using the available models.');
  paragraph(result.final_prediction.fusion?.policy || 'The backend decision rule determines the image verdict. See the diagnostic model scores below.');

  room(43);
  heading('Observations & limitations');
  paragraph('Scores are uncalibrated model outputs, not verified probabilities of authenticity. Grad-CAM shows model attention, not a confirmed manipulation boundary. Compression, image quality, and unfamiliar generators can affect results. Assess this report alongside the original source and context.', { size: 9 });
  paragraph('The JSON download contains the complete response, including every returned diagnostic field.', { size: 9 });

  for (const face of [result.final_prediction, ...(result.predictions || [])]) {
    newPage();
    heading(face.face_id > 0 ? `Face ${face.face_id} / specialist analysis` : 'Image assessment / specialist analysis');
    verdict(face.label, margin, y + 3, 20); y += 15;
    if (face.reason) paragraph(readable(face.reason), { size: 9, gap: 2 });
    line('Specialist model', margin + 3, y, 9, muted);
    line('Prediction', margin + 98, y, 9, muted);
    line('Fake score', margin + 142, y, 9, muted); y += 5;
    for (const [key, name] of modelNames) {
      if (key === 'authenticity' && !face.authenticity) continue;
      room(11);
      const model = face[key];
      const available = model && model.loaded !== false && model.score != null;
      pdf.setFillColor('#f0f3e9'); pdf.rect(margin, y, width, 10, 'F');
      line(name, margin + 3, y + 6.5, 9);
      line(available ? verdictName(model.label) : 'Unavailable', margin + 98, y + 6.5, 9);
      line(available ? percent(model.probabilities?.FAKE) : '-', margin + 142, y + 6.5, 9); y += 11;
    }
    y += 10;
    const visuals = [['Face crop', face.face_crop_base64], ['Grad-CAM attention', face.gradcam_heatmap_base64], ['Combined view', face.gradcam_overlay_base64]].filter(([, data]) => data);
    if (visuals.length) {
      room(66);
      visuals.forEach(([label, data], index) => {
        const x = margin + index * 60;
        pdf.setFillColor('#f0f3e9'); pdf.rect(x, y, 56, 51, 'F');
        addImage(data, x, y, 56, 51);
        line(label, x, y + 57, 9, muted);
      });
      y += 67;
    }
    if (face.box) paragraph(`Face bounds (x, y, width, height): ${face.box.join(', ')}.`, { size: 9, gap: 2 });
    if (face.fusion?.policy) paragraph(`Decision policy: ${face.fusion.policy}`, { size: 9, gap: 2 });
    if (face.combined_result?.votes) paragraph(`Supplementary votes: ${face.combined_result.votes.FAKE || 0} fake, ${face.combined_result.votes.REAL || 0} real. ${readable(face.combined_result.consensus)}.`, { size: 9 });
    const router = face.shadow_routing;
    if (router) {
      room(42); heading('Routing diagnostics');
      paragraph(`Image type: ${readable(router.image_type)}. Router score: ${percent(router.router_confidence)}.`, { size: 9, gap: 2 });
      paragraph(`Mode: ${router.shadow_mode ? 'Shadow (does not change the image verdict)' : 'Production'}. Policy: ${readable(router.routing_policy)}.`, { size: 9, gap: 2 });
      paragraph(`Selected specialists: ${router.selected_specialists?.join(', ') || 'Not available'}.`, { size: 9, gap: 2 });
    }
  }
  const total = pdf.getNumberOfPages();
  for (let page = 1; page <= total; page++) {
    pdf.setPage(page);
    line('DeepTrace / Image forensics', margin, 287, 8, muted);
    line(`${result.reportId} / ${page} of ${total}`, 193, 287, 8, muted, 'normal', { align: 'right' });
  }
  pdf.save(`${result.reportId}-DeepTrace.pdf`);
}
