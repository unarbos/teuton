/// <reference types="vitest/config" />
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

// In dev, proxy /api and /openapi.json to the FastAPI backend on :8767 so the
// SPA can call the real API without CORS. Override with VITE_API_PROXY.
const apiTarget = process.env.VITE_API_PROXY || 'http://127.0.0.1:8767';

export default defineConfig({
    plugins: [sveltekit()],
    server: {
        port: 5173,
        strictPort: false,
        proxy: {
            '/api': { target: apiTarget, changeOrigin: true, ws: false },
            '/healthz': { target: apiTarget, changeOrigin: true },
            '/openapi.json': { target: apiTarget, changeOrigin: true }
        }
    },
    test: {
        environment: 'jsdom',
        globals: true,
        include: ['tests/**/*.{test,spec}.ts'],
        setupFiles: ['./tests/setup.ts']
    }
});
