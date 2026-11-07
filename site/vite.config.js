import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads ../level-*/results*.json at build time, so every recorded number on the page is imported, never typed.
export default defineConfig({
  base: '/distributed-training-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
