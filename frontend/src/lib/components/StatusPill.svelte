<script lang="ts">
    import { cls } from '$lib/format';

    export let status: string;
    export let title: string = '';

    /** Map status -> ink-styling. Defaults to muted. */
    function variant(status: string): string {
        if (status === 'live') return 'bg-ink text-ink-inv';
        if (status === 'verified' || status === 'completed' || status === 'pass') return 'bg-ink text-ink-inv';
        if (status === 'failed' || status === 'fail') return 'opacity-70 italic';
        if (status === 'stale') return 'opacity-70';
        return 'opacity-55';
    }
</script>

<span
    class={cls(
        'inline-block mono text-[10px] uppercase tracking-widest px-[6px] py-[2px] border dashed whitespace-nowrap',
        variant(status)
    )}
    title={title || status}
>
    {status}
</span>
