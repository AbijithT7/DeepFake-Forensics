import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { include: ['jspdf'] },
  server: {
    port: 3000,
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
    },
  },
  preview: {
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
