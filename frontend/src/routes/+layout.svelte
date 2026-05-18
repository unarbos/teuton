<script lang="ts">
    import '../app.css';
    import { page } from '$app/stores';
    import RunPicker from '$lib/components/RunPicker.svelte';
    import ThemeToggle from '$lib/components/ThemeToggle.svelte';
    import { runs } from '$lib/stores/runs';
    import { snapshot } from '$lib/stores/snapshot';
    import { queue } from '$lib/stores/queue';

    let resolvedRun: string = '';

    // Keep stores in sync with the URL's run_id. When ?run_id changes (user
    // picks a run), restart both the snapshot poll and the SSE subscription.
    $: {
        const params = $page.url.searchParams;
        const requested = params.get('run_id') || '';
        const fallback = $runs.default_run_id || ($runs.runs[0] ?? '');
        resolvedRun = requested || fallback;
        snapshot.setRunId(resolvedRun || null);
        queue.setRunId(resolvedRun || null, 'train');
    }
</script>

<div class="mx-auto max-w-[1400px] px-6 md:px-8 py-6 md:py-8 flex flex-col gap-6">
    <header class="flex items-center justify-end gap-3 border-b dashed pb-2 text-[12px] uppercase">
        <RunPicker selected={resolvedRun} />
        <ThemeToggle />
    </header>
    <slot />
</div>
