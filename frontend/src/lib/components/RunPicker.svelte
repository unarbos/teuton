<script lang="ts">
    import { goto } from '$app/navigation';
    import { page } from '$app/stores';
    import { onMount } from 'svelte';
    import { runs } from '$lib/stores/runs';

    export let selected: string = '';

    onMount(() => {
        runs.refresh();
    });

    function handle(ev: Event) {
        const value = (ev.target as HTMLSelectElement).value;
        const url = new URL($page.url);
        if (value) {
            url.searchParams.set('run_id', value);
        } else {
            url.searchParams.delete('run_id');
        }
        goto(`${url.pathname}${url.search}`, { replaceState: false, keepFocus: true });
    }
</script>

<label class="mono text-[11px] tracking-wider uppercase flex items-center gap-2">
    <span class="opacity-60">Run</span>
    <select
        class="bg-paper text-ink border dashed px-2 py-1 mono text-[11px] uppercase tracking-wider focus:outline-none focus:border-solid"
        value={selected}
        on:change={handle}
    >
        {#if !$runs.runs.length}
            <option value="">{$runs.loading ? 'loading...' : '--'}</option>
        {/if}
        {#if selected && !$runs.runs.includes(selected)}
            <option value={selected}>{selected}</option>
        {/if}
        {#each $runs.runs as r}
            <option value={r}>{r}</option>
        {/each}
    </select>
</label>
