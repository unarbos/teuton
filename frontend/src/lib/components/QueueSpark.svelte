<script lang="ts">
    import { onDestroy, onMount } from 'svelte';
    import uPlot from 'uplot';
    import type { QueueHistoryPoint } from '$lib/api/types';

    export let history: QueueHistoryPoint[] = [];
    export let height: number = 80;

    let container: HTMLDivElement;
    let plot: uPlot | null = null;
    let observer: ResizeObserver | null = null;

    function buildOpts(width: number): uPlot.Options {
        return {
            width,
            height,
            padding: [4, 4, 4, 4],
            cursor: { drag: { x: false, y: false }, sync: { key: 'queue-spark' } },
            legend: { show: false },
            scales: { x: { time: true }, y: { range: (_self, _min, max) => [0, Math.max(1, max)] } },
            axes: [
                { show: false },
                { show: false }
            ],
            series: [
                {},
                {
                    label: 'depth',
                    stroke: 'rgb(var(--ink))',
                    width: 1.5,
                    fill: 'rgba(var(--ink), 0.06)',
                    points: { show: false }
                }
            ]
        };
    }

    function toData(points: QueueHistoryPoint[]): uPlot.AlignedData {
        // uPlot requires the X axis to be sorted ascending; the ring buffer
        // already enforces that, so we just split into two parallel arrays.
        const xs: number[] = new Array(points.length);
        const ys: number[] = new Array(points.length);
        for (let i = 0; i < points.length; i++) {
            xs[i] = points[i].ts;
            ys[i] = points[i].depth_total;
        }
        return [xs, ys];
    }

    onMount(() => {
        const data = toData(history);
        if (data[0].length === 0) data[0] = [0]; // uPlot needs at least 1 point
        if (data[1].length === 0) data[1] = [0];
        plot = new uPlot(buildOpts(container.clientWidth || 600), data, container);
        observer = new ResizeObserver(() => {
            if (plot) plot.setSize({ width: container.clientWidth, height });
        });
        observer.observe(container);
    });

    onDestroy(() => {
        observer?.disconnect();
        plot?.destroy();
    });

    $: if (plot && history) {
        const data = toData(history);
        if (data[0].length > 0) plot.setData(data);
    }
</script>

<div class="w-full" bind:this={container} style:height={`${height}px`}></div>
