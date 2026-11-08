import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/ml-coding-lab/',
  define: {
    // Cache-buster for files the Python worker fetches (public/py/*.py).
    __BUILD_ID__: JSON.stringify(Date.now().toString(36)),
  },
  optimizeDeps: {
    exclude: ['pyodide']
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
