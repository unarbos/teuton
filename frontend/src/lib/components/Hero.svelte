<script lang="ts">
    import { fmtAgeLabel, shortHotkey } from '$lib/format';
    import type { SnapshotResponse } from '$lib/api/types';

    export let snapshot: SnapshotResponse | null = null;
    export let lastUpdatedUnix: number = 0;
    export let sseConnected: boolean = false;

    $: meta = snapshot?.meta;
    $: chain = meta?.health?.chain;
    $: bucketScan = meta?.health?.states?.bucket?.updated_unix ?? null;
    $: chainScan = meta?.health?.states?.chain?.updated_unix ?? null;

    function pillBits(): string[] {
        if (!meta) return [];
        const now = lastUpdatedUnix || meta.generated_unix;
        const bits: string[] = [];
        bits.push(`NETUID ${meta.netuid}`);
        if (meta.bucket) bits.push(`BUCKET ${meta.bucket}`);
        if (meta.run_id) bits.push(`RUN ${shortRun(meta.run_id)}`);
        if (chain?.current_block != null) bits.push(`BLOCK ${chain.current_block}`);
        if (bucketScan) bits.push(`BUCKET SCAN ${fmtAgeLabel(Math.max(0, now - bucketScan))}`);
        if (chainScan) bits.push(`CHAIN SCAN ${fmtAgeLabel(Math.max(0, now - chainScan))}`);
        bits.push(`SSE ${sseConnected ? 'LIVE' : 'OFFLINE'}`);
        return bits;
    }

    function shortRun(run: string): string {
        return run.length > 36 ? `${run.slice(0, 36)}\u2026` : run;
    }
</script>

<div class="text-center mb-2">
    <div class="text-[20px] font-bold tracking-[0.1em] uppercase">Teutonic</div>
    <div class="text-[11px] mt-1 opacity-75 mono uppercase tracking-wider flex flex-wrap justify-center gap-2">
        {#each pillBits() as bit}
            <span>{bit}</span>
            <span class="opacity-30">|</span>
        {/each}
    </div>
</div>
