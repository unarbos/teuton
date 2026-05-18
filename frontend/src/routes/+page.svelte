<script lang="ts">
    import CompletedTable from '$lib/components/CompletedTable.svelte';
    import ComputeChart from '$lib/components/ComputeChart.svelte';
    import Hero from '$lib/components/Hero.svelte';
    import MinersTable from '$lib/components/MinersTable.svelte';
    import OutstandingTable from '$lib/components/OutstandingTable.svelte';
    import QueuePanel from '$lib/components/QueuePanel.svelte';
    import { queue } from '$lib/stores/queue';
    import { snapshot } from '$lib/stores/snapshot';

    $: snap = $snapshot.snapshot;
    $: queueSnap = $queue.snapshot ?? snap?.queue ?? null;
    $: outstanding = queueSnap?.outstanding ?? snap?.jobs.outstanding ?? [];
    $: completed = snap?.jobs.completed ?? [];
    $: machines = snap?.machines ?? [];
</script>

<Hero snapshot={snap} lastUpdatedUnix={$snapshot.last_updated_unix} sseConnected={$queue.connected} />

{#if $snapshot.error && !snap}
    <p class="mono text-[11px] border dashed px-3 py-2 text-warn">ERROR: {$snapshot.error}</p>
{/if}

<QueuePanel snapshot={queueSnap} sseConnected={$queue.connected} />

<ComputeChart completed={completed} />

<MinersTable machines={machines} />

<OutstandingTable rows={outstanding.map((entry) => ({
    job_id: entry.job_id,
    kind: ('kind' in entry ? (entry as { kind: string }).kind : '') || jobKind(entry.job_id),
    assigned_hotkey: entry.assigned_hotkey,
    assigned_worker: entry.assigned_worker,
    attempt: entry.attempt || 0,
    created_unix: entry.created_unix || 0,
    deadline_unix: entry.deadline_unix || 0,
    age_sec: null,
    deadline_sec: null,
    manifest_uri: 'manifest_uri' in entry ? (entry as { manifest_uri: string }).manifest_uri : null,
    grant_uri: 'grant_uri' in entry ? (entry as { grant_uri: string | null }).grant_uri : null,
    role: 'role' in entry ? (entry as { role: string }).role : 'train'
}))} />

<CompletedTable rows={completed} />

<script context="module" lang="ts">
    export function jobKind(jobId: string): string {
        for (const suffix of ['-fwd', '-bwd', '-outer', '-reduce', '-inner', '-eval']) {
            if (jobId.endsWith(suffix)) {
                const base = suffix.slice(1);
                return ['fwd', 'bwd', 'outer'].includes(base) ? `pipe_${base}` : base;
            }
        }
        if (jobId.startsWith('audit-')) return 'audit_replay';
        return '';
    }
</script>
