/**
 * Typed EventSource wrapper with exponential reconnect.
 *
 * The browser's native ``EventSource`` already auto-reconnects on transport
 * errors, but it doesn't expose a "we are now connected" signal or surface
 * structured parse errors. This wrapper closes the gap so the Svelte store
 * driving the queue panel can show an honest connection indicator and
 * back off cleanly when the backend is temporarily unreachable.
 */
import type { QueueSnapshot } from './types';

export type QueueStreamMessage =
    | { kind: 'snapshot'; snapshot: QueueSnapshot }
    | { kind: 'open' }
    | { kind: 'error'; error: string };

export interface QueueStreamOptions {
    runId: string;
    role?: 'train' | 'audit';
    onMessage: (msg: QueueStreamMessage) => void;
    /** Base retry delay (ms); capped at ``maxRetryMs``. Defaults to 500ms / 30s. */
    baseRetryMs?: number;
    maxRetryMs?: number;
}

export class QueueStream {
    private es: EventSource | null = null;
    private retryMs: number;
    private readonly base: number;
    private readonly max: number;
    private stopped = false;
    private timer: ReturnType<typeof setTimeout> | null = null;

    constructor(private readonly opts: QueueStreamOptions) {
        this.base = opts.baseRetryMs ?? 500;
        this.max = opts.maxRetryMs ?? 30_000;
        this.retryMs = this.base;
    }

    start(): void {
        if (this.stopped) return;
        const role = this.opts.role ?? 'train';
        const url = `/api/queue/stream?run_id=${encodeURIComponent(this.opts.runId)}&role=${role}`;
        const es = new EventSource(url, { withCredentials: false });
        this.es = es;

        es.addEventListener('open', () => {
            this.retryMs = this.base; // reset backoff on successful connect
            this.opts.onMessage({ kind: 'open' });
        });

        // The backend emits ``event: queue\ndata: {...}`` so we listen on the
        // named event AND the default 'message' channel (some proxies strip
        // event names).
        const handle = (ev: MessageEvent) => {
            try {
                const snapshot = JSON.parse(ev.data) as QueueSnapshot;
                this.opts.onMessage({ kind: 'snapshot', snapshot });
            } catch (err) {
                this.opts.onMessage({ kind: 'error', error: `parse error: ${(err as Error).message}` });
            }
        };
        es.addEventListener('queue', handle as EventListener);
        es.addEventListener('message', handle as EventListener);

        es.addEventListener('error', () => {
            this.opts.onMessage({ kind: 'error', error: `transport error, retrying in ${this.retryMs}ms` });
            es.close();
            this.es = null;
            if (this.stopped) return;
            this.timer = setTimeout(() => this.start(), this.retryMs);
            this.retryMs = Math.min(this.max, this.retryMs * 2);
        });
    }

    stop(): void {
        this.stopped = true;
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
        if (this.es) {
            this.es.close();
            this.es = null;
        }
    }
}
