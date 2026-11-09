const DATABASE = 'deeptrace-reports';
const STORE = 'reports';

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Report storage is blocked by another tab.'));
  });
}
async function transaction(mode, action) {
  const database = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE, mode);
    const request = action(tx.objectStore(STORE));
    tx.oncomplete = () => { database.close(); resolve(request?.result); };
    tx.onerror = tx.onabort = () => { database.close(); reject(tx.error); };
  });
}
export async function readReports() {
  let reports = await transaction('readonly', store => store.getAll());
  try {
    const previous = JSON.parse(localStorage.getItem('deeptrace_analysis_history') || '[]');
    if (Array.isArray(previous) && previous.length) {
      for (const item of previous.filter(item => item?.id && item?.fullResult)) await saveReport(item);
      localStorage.removeItem('deeptrace_analysis_history');
      reports = await transaction('readonly', store => store.getAll());
    }
  } catch { /* Malformed legacy storage must not prevent new reports from opening. */ }
  return reports.filter(item => item?.id && item?.fullResult).sort((a, b) => b.timestamp.localeCompare(a.timestamp));
}
export const saveReport = report => transaction('readwrite', store => store.put(report));
export const removeReport = id => transaction('readwrite', store => store.delete(id));
export const removeAllReports = () => transaction('readwrite', store => store.clear());
