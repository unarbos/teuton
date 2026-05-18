import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),
    kit: {
        adapter: adapter({
            // FastAPI's StaticFiles mount looks for index.html and a SPA fallback,
            // so we emit a fully self-contained SPA bundle.
            pages: 'build',
            assets: 'build',
            fallback: 'index.html',
            precompress: false,
            strict: false
        }),
        alias: {
            $lib: 'src/lib'
        }
    }
};

export default config;
