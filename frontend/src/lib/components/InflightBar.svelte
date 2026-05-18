<script lang="ts">
    import { cls } from '$lib/format';

    export let depth: number;
    export let cap: number;
    export let atCap: boolean = false;
    /** When true, render a compact (inline) variant for inside table cells. */
    export let compact: boolean = false;

    $: pct = cap > 0 ? Math.min(100, (depth / cap) * 100) : 0;
</script>

{#if cap > 0}
    <span
        class={cls(
            'inline-flex items-center gap-2 mono text-[11px]',
            atCap && 'font-bold',
            compact ? 'whitespace-nowrap' : ''
        )}
    >
        <span
            class={cls('relative border dotted-faint', compact ? 'w-[56px] h-[6px]' : 'flex-1 h-[8px]')}
            aria-hidden="true"
        >
            <span
                class={cls('block h-full bg-ink', atCap ? 'opacity-100' : 'opacity-55')}
                style:width={`${pct.toFixed(0)}%`}
            ></span>
        </span>
        <span class={compact ? 'tabular-nums' : 'tabular-nums w-[44px] text-right'}>
            {depth}/{cap}
        </span>
    </span>
{:else}
    <span class="mono text-[11px] tabular-nums">{depth}</span>
{/if}
