/**
 * Smoke test for the snapshot store's fetch + reconcile cycle. Uses
 * vi.spyOn on global fetch to avoid hitting the network.
 */
import { afterEach, describe, expect, test, vi } from 'vitest';
import { get } from 'svelte/store';
import { snapshot } from '../src/lib/stores/snapshot';
import type { SnapshotResponse } from '../src/lib/api/types';

const stubResponse: SnapshotResponse = {
    meta: {
        bucket: 'b',
        netuid: 0,
        run_id: 'r',
        generated_unix: 100,
        max_jobs: 50,
        max_inflight_per_hotkey: 4,
        heartbeat_ttl_sec: 30,
        source: 'sqlite',
        health: null
    },
    run: { run_id: 'r' },
    queue: null,
    audit_queue: null,
    machines: [],
    jobs: { outstanding: [], completed: [], audit_outstanding: [] }
};

afterEach(() => {
    snapshot.stop();
    vi.restoreAllMocks();
});

describe('snapshot store', () => {
    test('refresh loads the snapshot into the store', async () => {
        vi.spyOn(globalThis, 'fetch').mockResolvedValue(
            new Response(JSON.stringify(stubResponse), {
                status: 200,
                headers: { 'content-type': 'application/json' }
            }) as unknown as Response
        );
        snapshot.start({ runId: 'r', intervalMs: 60_000 });
        // Allow microtask to resolve.
        await new Promise((r) => setTimeout(r, 10));
        const state = get(snapshot);
        expect(state.snapshot).not.toBeNull();
        expect(state.snapshot?.meta.run_id).toBe('r');
        expect(state.error).toBeNull();
    });

    test('http error populates error field', async () => {
        vi.spyOn(globalThis, 'fetch').mockResolvedValue(
            new Response('boom', { status: 500, statusText: 'oops' }) as unknown as Response
        );
        snapshot.start({ runId: 'r', intervalMs: 60_000 });
        await new Promise((r) => setTimeout(r, 10));
        const state = get(snapshot);
        expect(state.error).toMatch(/500/);
    });
});
