import '@testing-library/jest-dom';
import { render } from '@testing-library/svelte';
import { afterEach, beforeAll, describe, expect, test, vi } from 'vitest';

// jsdom doesn't ship matchMedia or ResizeObserver; uPlot needs both. Stub
// them BEFORE QueuePanel (which transitively imports uPlot) is evaluated.
beforeAll(() => {
    if (!('matchMedia' in window)) {
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
    if (!('ResizeObserver' in globalThis)) {
        class FakeResizeObserver {
            observe() {}
            unobserve() {}
            disconnect() {}
        }
        (globalThis as { ResizeObserver?: typeof ResizeObserver }).ResizeObserver =
            FakeResizeObserver as unknown as typeof ResizeObserver;
    }
});

const QueuePanel = (await import('../src/lib/components/QueuePanel.svelte')).default;
type QueueSnapshot = typeof import('../src/lib/api/types')['QueueSnapshot'] extends never
    ? never
    : import('../src/lib/api/types').QueueSnapshot;

const sample: QueueSnapshot = {
    run_id: 'run-x',
    role: 'train',
    snapshot_unix: Math.floor(Date.now() / 1000) - 5,
    snapshot_id: 99,
    depth_total: 6,
    depth_by_hotkey: { '5GabcDEF12345': 4, '5GfedCBA67890': 2 },
    max_inflight_per_hotkey: 4,
    at_cap_count: 1,
    at_cap_hotkeys: ['5GabcDEF12345'],
    oldest_entry_age_sec: 12,
    oldest_job_id: 'j-e1234-s0-mb1-fwd',
    outstanding: [],
    history: [
        { ts: Math.floor(Date.now() / 1000) - 60, depth_total: 3, at_cap_count: 0 },
        { ts: Math.floor(Date.now() / 1000) - 30, depth_total: 5, at_cap_count: 1 },
        { ts: Math.floor(Date.now() / 1000), depth_total: 6, at_cap_count: 1 }
    ]
};

afterEach(() => {
    vi.restoreAllMocks();
});

describe('QueuePanel', () => {
    test('renders depth headline + per-miner bars', () => {
        const { container, getByText } = render(QueuePanel, {
            props: { snapshot: sample, sseConnected: true }
        });
        expect(getByText(/6 \/ 8/)).toBeInTheDocument(); // depth / (cap*miners)
        expect(getByText(/BACKPRESSURE/)).toBeInTheDocument();
        expect(container.querySelector('.mono')).not.toBeNull();
        // Two miner bars rendered.
        const bars = container.querySelectorAll('[title="5GabcDEF12345"], [title="5GfedCBA67890"]');
        expect(bars.length).toBe(2);
    });

    test('renders empty state when snapshot missing', () => {
        const { getByText } = render(QueuePanel, { props: { snapshot: null, sseConnected: false } });
        expect(getByText(/Awaiting first queue snapshot/i)).toBeInTheDocument();
    });
});
