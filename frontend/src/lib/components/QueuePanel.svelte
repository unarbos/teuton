<script lang="ts">
    import QueueSpark from './QueueSpark.svelte';
    import { cls, fmtDurationSec, shortHotkey } from '$lib/format';
    import type { QueueSnapshot } from '$lib/api/types';

    export let snapshot: QueueSnapshot | null = null;
    export let sseConnected: boolean = false;

    $: cap = snapshot?.max_inflight_per_hotkey ?? 0;
    $: depth = snapshot?.depth_total ?? 0;
    $: minerCount = Object.keys(snapshot?.depth_by_hotkey ?? {}).length;
    $: maxNet = cap > 0 ? cap * Math.max(minerCount, 1) : 0;
    $: atCap = snapshot?.at_cap_count ?? 0;
    $: backpressureFrac = minerCount > 0 ? atCap / minerCount : 0;

    function snapAge(snapshot_unix: number, now = Math.floor(Date.now() / 1000)): string {
        if (!snapshot_unix) return '--';
        return fmtDurationSec(Math.max(0, now - snapshot_unix));
    }

    function sortedHotkeys(): Array<[string, number]> {
        if (!snapshot) return [];
        return Object.entries(snapshot.depth_by_hotkey).sort((a, b) => b[1] - a[1]);
    }
</script>

<section>
    <div class="flex items-center justify-between border-b dashed pb-1 mb-2 mono text-[11px] uppercase tracking-widest font-bold">
        <span>Queue</span>
        <span class="font-normal opacity-60">
            {sseConnected ? 'live' : 'reconnecting'} ·
            id={snapshot?.snapshot_id ?? 0}
        </span>
    </div>

    {#if !snapshot}
        <p class="mono text-[11px] opacity-60 py-4 text-center uppercase">Awaiting first queue snapshot</p>
    {:else}
        <div class="flex flex-wrap gap-4 items-baseline border-b dotted-faint pb-2">
            <div>
                <div class="mono text-[28px] leading-none tracking-wider">
                    {depth}{maxNet > 0 ? ` / ${maxNet}` : ''}
                </div>
                <div class="mono text-[12px] uppercase tracking-wider opacity-55">
                    {cap > 0
                        ? `outstanding / cap (${cap} per miner · ${minerCount} miners)`
                        : 'outstanding entries'}
                </div>
            </div>
            <div class="mono text-[10px] leading-snug uppercase tracking-widest opacity-75 ml-auto">
                <div>
                    BACKPRESSURE
                    <strong class={cls('ml-1', backpressureFrac >= 0.5 ? 'text-warn' : '')}>
                        {(backpressureFrac * 100).toFixed(0)}% ({atCap}/{minerCount})
                    </strong>
                </div>
                <div>
                    OLDEST
                    <strong class="ml-1">
                        {snapshot.oldest_entry_age_sec != null
                            ? `${fmtDurationSec(snapshot.oldest_entry_age_sec)}${snapshot.oldest_job_id ? ` (${shortHotkey(snapshot.oldest_job_id, 8, 6)})` : ''}`
                            : '--'}
                    </strong>
                </div>
                <div>
                    SNAPSHOT <strong class="ml-1">{snapAge(snapshot.snapshot_unix)} ago</strong>
                </div>
            </div>
        </div>

        <div class="border-b dotted-faint py-2">
            <QueueSpark history={snapshot.history} height={80} />
            <div class="flex justify-between mono text-[10px] uppercase tracking-wider opacity-55 mt-1">
                <span>-30m</span>
                <span>now</span>
            </div>
        </div>

        {#if sortedHotkeys().length}
            <div class="grid gap-y-[6px] gap-x-4 pt-2"
                 style="grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));">
                {#each sortedHotkeys() as [hk, d]}
                    {@const isAtCap = cap > 0 && d >= cap}
                    {@const pct = cap > 0 ? Math.min(100, (d / cap) * 100) : Math.min(100, d * 10)}
                    <div class="flex items-center gap-2 mono text-[11px]" title={hk}>
                        <span class="opacity-85 w-[96px]">{shortHotkey(hk, 5, 5)}</span>
                        <span class="flex-1 h-[8px] border dotted-faint relative" aria-hidden="true">
                            <span
                                class={cls('block h-full bg-ink', isAtCap ? 'opacity-100' : 'opacity-55')}
                                style:width={`${pct.toFixed(0)}%`}
                            ></span>
                        </span>
                        <span class={cls('w-[44px] text-right tabular-nums', isAtCap && 'font-bold')}>
                            {d}{cap > 0 ? `/${cap}` : ''}
                        </span>
                    </div>
                {/each}
            </div>
        {:else}
            <p class="mono text-[11px] opacity-55 py-2">No entries.</p>
        {/if}
    {/if}
</section>
