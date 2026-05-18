<script lang="ts">
    import FilterBar from './FilterBar.svelte';
    import InflightBar from './InflightBar.svelte';
    import StatusPill from './StatusPill.svelte';
    import { cls, fmtDurationSec, fmtPoints, shortHotkey, shortWorker } from '$lib/format';
    import type { Machine, WorkerRow } from '$lib/api/types';

    export let machines: Machine[] = [];

    interface Row {
        machine: Machine;
        worker: WorkerRow;
    }

    const FILTERS = [
        { id: 'all', label: 'All' },
        { id: 'live', label: 'Live' },
        { id: 'stale', label: 'Stale' },
        { id: 'at-cap', label: 'At-cap' }
    ];
    const SORT_KEYS = ['status', 'identity', 'uid', 'emission', 'inflight', 'receipts', 'ping'] as const;
    type SortKey = (typeof SORT_KEYS)[number];

    let activeFilter = 'all';
    let sort: { key: SortKey; dir: 'asc' | 'desc' } = { key: 'inflight', dir: 'desc' };

    $: rows = flatten(machines);

    function flatten(ms: Machine[]): Row[] {
        const out: Row[] = [];
        for (const m of ms) for (const w of m.workers) out.push({ machine: m, worker: w });
        return out;
    }

    function counts(rows: Row[]): Record<string, number> {
        const c: Record<string, number> = { all: rows.length, live: 0, stale: 0, 'at-cap': 0 };
        for (const r of rows) {
            const s = r.worker.status;
            if (s in c) c[s] = (c[s] || 0) + 1;
            if (r.worker.at_cap) c['at-cap'] = (c['at-cap'] || 0) + 1;
        }
        return c;
    }

    function filtered(rows: Row[]): Row[] {
        if (activeFilter === 'all') return rows;
        if (activeFilter === 'at-cap') return rows.filter((r) => r.worker.at_cap);
        return rows.filter((r) => r.worker.status === activeFilter);
    }

    function sortValue(r: Row, key: SortKey): number | string {
        const w = r.worker;
        const cap = (w.worker.capabilities ?? {}) as Record<string, unknown>;
        const chain = w.chain;
        if (key === 'status') return w.status;
        if (key === 'identity') return `${(w.worker.hotkey_ss58 as string) || ''}|${(w.worker.worker_id as string) || ''}`;
        if (key === 'uid') return chain?.uid ?? -1;
        if (key === 'emission') return chain?.emission ?? -1;
        if (key === 'receipts') return Number(w.n_receipts || 0);
        if (key === 'inflight') return Number(w.queue_depth || 0);
        if (key === 'ping') {
            const v = cap.rtt_to_bucket_ms;
            return typeof v === 'number' ? v : Number.POSITIVE_INFINITY;
        }
        return Number(w.queue_depth || 0);
    }

    function sorted(rows: Row[]): Row[] {
        const factor = sort.dir === 'asc' ? 1 : -1;
        return rows.slice().sort((a, b) => {
            const av = sortValue(a, sort.key);
            const bv = sortValue(b, sort.key);
            if (typeof av === 'number' && typeof bv === 'number') {
                if (av !== bv) return (av - bv) * factor;
            } else {
                const cmp = String(av).localeCompare(String(bv));
                if (cmp !== 0) return cmp * factor;
            }
            return String(sortValue(a, 'identity')).localeCompare(String(sortValue(b, 'identity')));
        });
    }

    function flipSort(key: SortKey) {
        if (sort.key === key) {
            sort = { key, dir: sort.dir === 'asc' ? 'desc' : 'asc' };
        } else {
            sort = { key, dir: key === 'ping' || key === 'identity' || key === 'status' ? 'asc' : 'desc' };
        }
    }

    function identityCell(w: WorkerRow): { short: string; full: string } {
        const hk = (w.worker.hotkey_ss58 as string) || '';
        const host = (w.worker.host_id as string) || '';
        const wid = (w.worker.worker_id as string) || '';
        const short = `${shortHotkey(hk).replace('\u2026', '')}/${shortHotkey(host, 3, 2)}/${shortHotkey(wid, 3, 2)}`;
        return { short, full: `miner=${hk} host=${host} worker=${wid}` };
    }

    function gpuLabel(w: WorkerRow): string {
        const cap = (w.worker.capabilities ?? {}) as Record<string, unknown>;
        return (
            (cap.gpu_name as string) ||
            (cap.gpu_class as string) ||
            (w.worker.gpu_index != null ? `gpu${w.worker.gpu_index}` : '--')
        );
    }

    function pingLabel(w: WorkerRow): string {
        const cap = (w.worker.capabilities ?? {}) as Record<string, unknown>;
        const ms = cap.rtt_to_bucket_ms;
        return typeof ms === 'number' ? fmtDurationSec(ms / 1000) : '--';
    }
</script>

<section>
    <div class="flex items-center justify-between border-b dashed pb-1 mb-2">
        <span class="mono text-[11px] uppercase tracking-widest font-bold">Miners</span>
        <span class="mono text-[11px] opacity-60">
            {filtered(rows).length} of {rows.length}
        </span>
    </div>
    <FilterBar
        options={FILTERS}
        active={activeFilter}
        counts={counts(rows)}
        on:change={(e) => (activeFilter = e.detail)}
    />
    <div class="overflow-auto scroll-thin max-h-[60vh]">
        <table class="w-full text-[11px] crosshair" style="table-layout: fixed;">
            <colgroup>
                <col style="width: 4%" /><col style="width: 8%" /><col style="width: 18%" />
                <col style="width: 14%" /><col style="width: 6%" /><col style="width: 9%" />
                <col style="width: 18%" /><col style="width: 8%" /><col style="width: 8%" />
            </colgroup>
            <thead class="sticky top-0 bg-paper">
                <tr class="border-b dashed">
                    {#each [
                        { label: '#', key: null },
                        { label: 'Status', key: 'status' as const },
                        { label: 'M/H/W', key: 'identity' as const },
                        { label: 'GPU', key: null },
                        { label: 'UID', key: 'uid' as const },
                        { label: 'Emit', key: 'emission' as const },
                        { label: 'Inflight', key: 'inflight' as const },
                        { label: 'Rcpts', key: 'receipts' as const },
                        { label: 'Ping', key: 'ping' as const }
                    ] as h}
                        <th
                            class={cls(
                                'mono text-[10px] uppercase tracking-widest text-left pr-2 py-1',
                                h.key && 'cursor-pointer'
                            )}
                            on:click={() => h.key && flipSort(h.key)}
                        >
                            {h.label}{sort.key === h.key && h.key ? (sort.dir === 'asc' ? ' \u25B2' : ' \u25BC') : ''}
                        </th>
                    {/each}
                </tr>
            </thead>
            <tbody>
                {#each sorted(filtered(rows)) as r, i}
                    {@const ident = identityCell(r.worker)}
                    <tr class="border-b dotted-faint">
                        <td class="py-1">{i + 1}</td>
                        <td><StatusPill status={r.worker.status} /></td>
                        <td><code title={ident.full}>{ident.short}</code></td>
                        <td>{gpuLabel(r.worker)}</td>
                        <td class="text-right">{r.worker.chain?.uid ?? '--'}</td>
                        <td class="text-right" title={String(r.worker.chain?.emission ?? '')}>
                            {r.worker.chain?.emission != null ? fmtPoints(r.worker.chain.emission) : '--'}
                        </td>
                        <td class="text-right">
                            <InflightBar depth={r.worker.queue_depth} cap={r.worker.queue_cap} atCap={r.worker.at_cap} compact />
                        </td>
                        <td class="text-right">{r.worker.n_receipts}</td>
                        <td class="text-right">{pingLabel(r.worker)}</td>
                    </tr>
                {/each}
                {#if !rows.length}
                    <tr><td colspan="9" class="text-center py-4 opacity-60">No miners.</td></tr>
                {/if}
            </tbody>
        </table>
    </div>
</section>
