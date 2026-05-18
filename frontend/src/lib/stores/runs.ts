/**
 * One-shot load of the run list with manual refresh.
 *
 * The header's <RunPicker> consumes this; selecting a run updates the URL
 * (?run_id=...), which all other stores observe through SvelteKit's $page.
 */
import { writable } from 'svelte/store';
import { api, ApiError } from '$lib/api/client';
import type { RunsResponse } from '$lib/api/types';

export interface RunsState {
    loading: boolean;
    runs: string[];
    default_run_id: string;
    error: string | null;
}

const initial: RunsState = { loading: false, runs: [], default_run_id: '', error: null };

function createRunsStore() {
    const { subscribe, update, set } = writable<RunsState>(initial);

    async function refresh(): Promise<void> {
        update((s) => ({ ...s, loading: true, error: null }));
        try {
            const res: RunsResponse = await api.runs();
            set({ loading: false, runs: res.runs, default_run_id: res.default_run_id, error: null });
        } catch (err) {
            const msg = err instanceof ApiError ? err.message : String(err);
            update((s) => ({ ...s, loading: false, error: msg }));
        }
    }

    return { subscribe, refresh };
}

export const runs = createRunsStore();
