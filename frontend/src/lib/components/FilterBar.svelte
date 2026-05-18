<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    import { cls } from '$lib/format';

    export let options: { id: string; label: string }[];
    export let active: string;
    export let counts: Record<string, number> = {};

    const dispatch = createEventDispatcher<{ change: string }>();

    function pick(id: string) {
        if (id === active) return;
        dispatch('change', id);
    }
</script>

<div class="flex flex-wrap gap-[6px] mb-2">
    {#each options as opt}
        <button
            type="button"
            class={cls(
                'border dashed px-[10px] py-[4px] mono text-[11px] uppercase tracking-widest cursor-pointer hover:bg-ink hover:text-ink-inv',
                active === opt.id && 'bg-ink text-ink-inv'
            )}
            on:click={() => pick(opt.id)}
        >
            {opt.label}
            {#if counts[opt.id] != null}
                <span class={cls('ml-[6px] font-normal', active === opt.id ? 'opacity-90' : 'opacity-60')}
                    >{counts[opt.id]}</span
                >
            {/if}
        </button>
    {/each}
</div>
