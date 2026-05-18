/**
 * Polled /api/snapshot store.
 *
 * Drives the slower-changing parts of the dashboard (miners table, completed
 * jobs, hero meta). The queue panel and outstanding-jobs table also re-render
 * on snapshot updates -- but the SSE-fed queue store wins for sub-second
 * freshness.
 */
import { writable } from 'svelte/store';
import { api, ApiError } from '$lib/api/client';
import type { SnapshotResponse } from '$lib/api/types';

export interface SnapshotState {
    loading: boolean;
    last_updated_unix: number;
    snapshot: SnapshotResponse | null;
    error: string | null;
}

const initial: SnapshotState = {
    loading: false,
    last_updated_unix: 0,
    snapshot: null,
    error: null
};

interface PollOptions {
    runId: string | null;
    intervalMs?: number;
}

function createSnapshotStore() {
    const { subscribe, update, set } = writable<SnapshotState>(initial);
    let timer: ReturnType<typeof setTimeout> | null = null;
    let activeRunId: string | null = null;

    async function fetchOnce(runId: string | null): Promise<void> {
        update((s) => ({ ...s, loading: true }));
        try {
            const snap = await api.snapshot(runId || undefined);
            set({
                loading: false,
                last_updated_unix: Math.floor(Date.now() / 1000),
                snapshot: snap,
                error: null
            });
        } catch (err) {
            const msg = err instanceof ApiError ? err.message : String(err);
            update((s) => ({ ...s, loading: false, error: msg }));
        }
    }

    function start({ runId, intervalMs = 3000 }: PollOptions): void {
        stop();
        activeRunId = runId;
        // Fire-and-forget initial fetch. Each subsequent timer tick fires a
        // fresh fetch even if the previous one is still in flight; that's
        // fine -- we replace the whole state with the latest reply.
        void fetchOnce(runId);
        timer = setInterval(() => void fetchOnce(activeRunId), intervalMs);
    }

    function stop(): void {
        if (timer) {
            clearInterval(timer);
            timer = null;
        }
    }

    function setRunId(runId: string | null): void {
        if (runId === activeRunId) return;
        start({ runId });
    }

    return { subscribe, start, stop, setRunId, refresh: () => fetchOnce(activeRunId) };
}

export const snapshot = createSnapshotStore();
