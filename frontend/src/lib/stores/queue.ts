/**
 * SSE-fed queue store.
 *
 * Drives the live queue panel (depth, per-miner bars, sparkline) and the
 * outstanding-jobs table. Updates whenever the orchestrator advances
 * snapshot_id; reconnects automatically through ``QueueStream``.
 */
import { writable } from 'svelte/store';
import { api, ApiError } from '$lib/api/client';
import { QueueStream } from '$lib/api/sse';
import type { QueueSnapshot } from '$lib/api/types';

export interface QueueStoreState {
    snapshot: QueueSnapshot | null;
    connected: boolean;
    error: string | null;
}

const initial: QueueStoreState = { snapshot: null, connected: false, error: null };

function createQueueStore() {
    const { subscribe, update, set } = writable<QueueStoreState>(initial);
    let stream: QueueStream | null = null;
    let activeRunId: string | null = null;
    let activeRole: 'train' | 'audit' = 'train';

    async function bootstrap(runId: string, role: 'train' | 'audit'): Promise<void> {
        try {
            const res = await api.queue(runId, role);
            update((s) => ({ ...s, snapshot: res.queue ?? s.snapshot, error: null }));
        } catch (err) {
            const msg = err instanceof ApiError ? err.message : String(err);
            update((s) => ({ ...s, error: msg }));
        }
    }

    function start(runId: string, role: 'train' | 'audit' = 'train'): void {
        stop();
        activeRunId = runId;
        activeRole = role;
        if (!runId) {
            set(initial);
            return;
        }
        // Seed via REST first so the panel has something to render before SSE
        // wakes up.
        void bootstrap(runId, role);
        stream = new QueueStream({
            runId,
            role,
            onMessage: (msg) => {
                if (msg.kind === 'snapshot') {
                    update((s) => ({ ...s, snapshot: msg.snapshot, error: null }));
                } else if (msg.kind === 'open') {
                    update((s) => ({ ...s, connected: true }));
                } else if (msg.kind === 'error') {
                    update((s) => ({ ...s, connected: false, error: msg.error }));
                }
            }
        });
        stream.start();
    }

    function stop(): void {
        if (stream) {
            stream.stop();
            stream = null;
        }
        set({ snapshot: null, connected: false, error: null });
    }

    function setRunId(runId: string | null, role: 'train' | 'audit' = activeRole): void {
        if (runId === activeRunId && role === activeRole) return;
        if (!runId) {
            stop();
            return;
        }
        start(runId, role);
    }

    return { subscribe, start, stop, setRunId };
}

export const queue = createQueueStore();
