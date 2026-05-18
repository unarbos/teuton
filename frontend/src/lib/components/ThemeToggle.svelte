<script lang="ts">
    import { onMount } from 'svelte';

    let theme: 'light' | 'dark' = 'light';

    onMount(() => {
        const current = (document.documentElement.dataset.theme as 'light' | 'dark') || 'light';
        theme = current;
    });

    function toggle() {
        theme = theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = theme;
        try {
            localStorage.setItem('theme', theme);
        } catch (_) {
            /* swallow */
        }
    }
</script>

<button
    type="button"
    class="border dashed px-2 py-1 mono text-[11px] tracking-wider uppercase hover:bg-ink hover:text-ink-inv"
    on:click={toggle}
    aria-label="Toggle theme"
>
    {theme === 'dark' ? 'LIGHT' : 'DARK'}
</button>
