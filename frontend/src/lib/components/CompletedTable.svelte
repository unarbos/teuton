<script lang="ts">
    import FilterBar from './FilterBar.svelte';
    import StatusPill from './StatusPill.svelte';
    import { cls, fmtAgeLabel, fmtBytes, fmtDurationSec, fmtTime, shortWorker } from '$lib/format';
    import type { CompletedJobRow } from '$lib/api/types';

    export let rows: CompletedJobRow[] = [];

    const FILTERS = [
        { id: 'all', label: 'All' },
        { id: 'completed', label: 'Completed' },
        { id: 'verified', label: 'Verified' },
        { id: 'failed', label: 'Failed' }
    ];

    let activeFilter = 'all';

    function counts(rows: CompletedJobRow[]): Record<string, number> {
        const c: Record<string, number> = { all: rows.length };
        for (const r of rows) c[r.status] = (c[r.status] || 0) + 1;
        return c;
    }

    function filtered(rows: CompletedJobRow[]): CompletedJobRow[] {
        if (activeFilter === 'all') return rows;
        return rows.filter((r) => r.status === activeFilter);
    }
</script>

<section>
    <div class="flex items-center justify-between border-b dashed pb-1 mb-2">
        <span class="mono text-[11px] uppercase tracking-widest font-bold">Completed</span>
        <span class="mono text-[11px] opacity-60">
            {filtered(rows).length} of {rows.length} completed
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
                <col style="width: 4%" /><col style="width: 20%" /><col style="width: 12%" />
                <col style="width: 12%" /><col style="width: 13%" /><col style="width: 11%" />
                <col style="width: 24%" />
            </colgroup>
            <thead class="sticky top-0 bg-paper">
                <tr class="border-b dashed">
                    {#each ['#', 'Kind', 'Status', 'Miner', 'Finished', 'Latency', 'I/O'] as h, i}
                        <th class={cls('mono text-[10px] uppercase tracking-widest pr-2 py-1', i >= 4 ? 'text-right' : 'text-left')}>
                            {h}
                        </th>
                    {/each}
                </tr>
            </thead>
            <tbody>
                {#each filtered(rows) as r, i}
                    {@const ioBytes = (r.bytes_read || 0) + (r.bytes_written || 0)}
                    <tr class="border-b dotted-faint">
                        <td class="py-1">{i + 1}</td>
                        <td>{r.kind || '--'}</td>
                        <td><StatusPill status={r.status} /></td>
                        <td>
                            <code title={r.assigned_hotkey ?? ''}
                                >{shortWorker(r.assigned_worker || r.assigned_hotkey)}</code
                            >
                        </td>
                        <td class="text-right" title={fmtTime(r.finished_unix)}>
                            {r.finished_unix ? fmtAgeLabel(Math.max(0, Math.floor(Date.now() / 1000) - r.finished_unix)) : '--'}
                        </td>
                        <td class="text-right">{r.duration_sec != null ? fmtDurationSec(r.duration_sec) : '--'}</td>
                        <td class="text-right" title={`read ${fmtBytes(r.bytes_read)} / wrote ${fmtBytes(r.bytes_written)}`}>
                            {ioBytes > 0 ? fmtBytes(ioBytes) : '--'}
                        </td>
                    </tr>
                {/each}
                {#if !filtered(rows).length}
                    <tr><td colspan="7" class="text-center py-4 opacity-60">No completed jobs in window.</td></tr>
                {/if}
            </tbody>
        </table>
    </div>
</section>
