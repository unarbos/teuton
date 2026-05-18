/**
 * Thin typed fetch wrapper around the dashboard's REST surface.
 *
 * All endpoints honor an optional ``run_id`` query param; routes that don't
 * accept it just ignore the extra key. ``fetchJSON`` always rejects on
 * non-2xx so callers don't need to repeat error handling.
 */
import type {
    HealthResponse,
    QueueResponse,
    RunsResponse,
    SnapshotResponse
} from './types';

export class ApiError extends Error {
    constructor(public status: number, message: string) {
        super(message);
        this.name = 'ApiError';
    }
}

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(path, { credentials: 'same-origin', ...init });
    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new ApiError(res.status, `${res.status} ${res.statusText} ${text.slice(0, 200)}`);
    }
    return (await res.json()) as T;
}

function qs(params: Record<string, string | number | undefined | null>): string {
    const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '');
    if (!entries.length) return '';
    return '?' + new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
}

export const api = {
    health(): Promise<HealthResponse> {
        return fetchJSON('/healthz');
    },
    runs(): Promise<RunsResponse> {
        return fetchJSON('/api/runs');
    },
    snapshot(runId?: string): Promise<SnapshotResponse> {
        return fetchJSON(`/api/snapshot${qs({ run_id: runId })}`);
    },
    queue(runId?: string, role: 'train' | 'audit' = 'train'): Promise<QueueResponse> {
        return fetchJSON(`/api/queue${qs({ run_id: runId, role })}`);
    }
};

export type { HealthResponse, QueueResponse, RunsResponse, SnapshotResponse };
