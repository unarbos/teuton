<script lang="ts">
    import { cls, fmtDurationSec, shortHotkey, shortWorker } from '$lib/format';
    import type { OutstandingJobRow } from '$lib/api/types';

    export let rows: OutstandingJobRow[] = [];

    let nowUnix = Math.floor(Date.now() / 1000);
    setInterval(() => (nowUnix = Math.floor(Date.now() / 1000)), 1000);
</script>

<section>
    <div class="flex items-center justify-between border-b dashed pb-1 mb-2">
        <span class="mono text-[11px] uppercase tracking-widest font-bold">Outstanding</span>
        <span class="mono text-[11px] opacity-60">{rows.length} outstanding</span>
    </div>
    <div class="overflow-auto scroll-thin max-h-[60vh]">
        <table class="w-full text-[11px] crosshair" style="table-layout: fixed;">
            <colgroup>
                <col style="width: 4%" /><col style="width: 24%" /><col style="width: 14%" />
                <col style="width: 14%" /><col style="width: 14%" /><col style="width: 30%" />
            </colgroup>
            <thead class="sticky top-0 bg-paper">
                <tr class="border-b dashed">
                    {#each ['#', 'Kind', 'Miner', 'Age', 'Attempt', 'Deadline'] as h, i}
                        <th
                            class={cls(
                                'mono text-[10px] uppercase tracking-widest pr-2 py-1',
                                i === 0 ? 'text-left' : i >= 3 ? 'text-right' : 'text-left'
                            )}
                        >
                            {h}
                        </th>
                    {/each}
                </tr>
            </thead>
            <tbody>
                {#each rows as r, i}
                    {@const worker = r.assigned_worker || r.assigned_hotkey || '--'}
                    {@const age = r.created_unix ? Math.max(0, nowUnix - r.created_unix) : null}
                    {@const deadlineLeft = r.deadline_unix ? r.deadline_unix - nowUnix : null}
                    <tr class="border-b dotted-faint">
                        <td class="py-1">{i + 1}</td>
                        <td>{r.kind || '--'}</td>
                        <td>
                            <code title={`miner=${r.assigned_hotkey} worker=${r.assigned_worker ?? '--'}`}
                                >{shortWorker(worker)}</code
                            >
                        </td>
                        <td class="text-right">{age != null ? fmtDurationSec(age) : '--'}</td>
                        <td class={cls('text-right', r.attempt > 0 && 'font-bold')}>{r.attempt}</td>
                        <td class={cls('text-right', deadlineLeft != null && deadlineLeft < 0 && 'text-warn font-bold')}>
                            {deadlineLeft == null
                                ? '--'
                                : deadlineLeft < 0
                                  ? `-${fmtDurationSec(-deadlineLeft)}`
                                  : fmtDurationSec(deadlineLeft)}
                        </td>
                    </tr>
                {/each}
                {#if !rows.length}
                    <tr><td colspan="6" class="text-center py-4 opacity-60">Queue is empty.</td></tr>
                {/if}
            </tbody>
        </table>
    </div>
</section>
