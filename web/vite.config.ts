import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API (`uv run amber serve`) owns /api and /thumbs; the dev server forwards them.
const API = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': API, '/thumbs': API },
  },
})
