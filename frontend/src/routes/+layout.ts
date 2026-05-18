// Pure SPA: no SSR, no prerender. FastAPI's StaticFiles serves the built
// shell directly.
export const ssr = false;
export const prerender = false;
export const trailingSlash = 'never';
