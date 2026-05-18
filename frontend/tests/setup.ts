/**
 * Vitest global setup: shims uPlot's runtime requirements (matchMedia,
 * ResizeObserver) so components that wrap it can render under jsdom.
 */
import '@testing-library/jest-dom';

if (typeof window !== 'undefined' && !('matchMedia' in window)) {
    Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: (query: string) => ({
            matches: false,
            media: query,
            onchange: null,
            addListener: () => {},
            removeListener: () => {},
            addEventListener: () => {},
            removeEventListener: () => {},
            dispatchEvent: () => false
        })
    });
}

if (typeof globalThis.ResizeObserver === 'undefined') {
    class FakeResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
    }
    (globalThis as { ResizeObserver?: typeof ResizeObserver }).ResizeObserver =
        FakeResizeObserver as unknown as typeof ResizeObserver;
}
