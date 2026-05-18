<script lang="ts">
    import { onDestroy, onMount } from 'svelte';
    import uPlot from 'uplot';
    import { cls, fmtBytes, fmtDurationSec } from '$lib/format';
    import type { CompletedJobRow } from '$lib/api/types';

    export let completed: CompletedJobRow[] = [];
    export let windowSec: number = 30 * 60;
    export let binSec: number = 30;

    type MetricId = 'compute' | 'bandwidth' | 'jobs' | 'latency';

    const METRICS: { id: MetricId; label: string; unit: string }[] = [
        { id: 'compute', label: 'Compute', unit: 'CU/s' },
        { id: 'bandwidth', label: 'Bandwidth', unit: 'B/s' },
        { id: 'jobs', label: 'Jobs', unit: 'jobs/s' },
        { id: 'latency', label: 'Latency', unit: 'avg s' }
    ];

    let metric: MetricId = 'compute';
    let container: HTMLDivElement;
    let plot: uPlot | null = null;
    let observer: ResizeObserver | null = null;
    let nowUnix = Math.floor(Date.now() / 1000);
    let chartHeight = 160;

    function series(metric: MetricId, jobs: CompletedJobRow[]): { xs: number[]; ys: number[] } {
        const nBins = Math.max(1, Math.floor(windowSec / binSec));
        const totals = new Array<number>(nBins).fill(0);
        const counts = new Array<number>(nBins).fill(0);
        for (const j of jobs) {
            const finished = Number(j.finished_unix);
            if (!finished) continue;
            const age = nowUnix - finished;
            if (age < 0 || age >= windowSec) continue;
            const idx = nBins - 1 - Math.floor(age / binSec);
            if (idx < 0 || idx >= nBins) continue;
            let value = 0;
            if (metric === 'bandwidth') value = Number(j.bytes_read || 0) + Number(j.bytes_written || 0);
            else if (metric === 'jobs') value = 1;
            else if (metric === 'latency') value = Number(j.duration_sec || 0);
            else value = Number(j.compute_sec || 0);
            if (!Number.isFinite(value) || value <= 0) continue;
            totals[idx] += value;
            counts[idx] += 1;
        }
        const xs = new Array<number>(nBins);
        const ys = new Array<number>(nBins);
        for (let i = 0; i < nBins; i++) {
            xs[i] = nowUnix - (nBins - 1 - i) * binSec;
            if (metric === 'latency') {
                ys[i] = counts[i] > 0 ? totals[i] / counts[i] : 0;
            } else {
                ys[i] = totals[i] / binSec;
            }
        }
        return { xs, ys };
    }

    function fmtValue(m: MetricId, v: number): string {
        if (!Number.isFinite(v)) return '--';
        if (m === 'bandwidth') return `${fmtBytes(v)}/s`;
        if (m === 'jobs') return `${v.toFixed(v >= 1 ? 2 : 3)}/s`;
        if (m === 'latency') return fmtDurationSec(v);
        return `${v.toFixed(2)} CU/s`;
    }

    function metricLabel(): string {
        return METRICS.find((m) => m.id === metric)?.label ?? '';
    }

    function buildOpts(width: number, m: MetricId): uPlot.Options {
        return {
            width,
            height: chartHeight,
            padding: [12, 4, 24, 4],
            cursor: { drag: { x: false, y: false }, sync: { key: 'compute' } },
            legend: { show: false },
            scales: {
                x: { time: true },
                y: { range: (_self, _min, max) => [0, Math.max(1, max || 1)] }
            },
            axes: [
                {
                    stroke: 'rgb(var(--ink-muted))',
                    grid: { show: false },
                    ticks: { show: false }
                },
                {
                    stroke: 'rgb(var(--ink-muted))',
                    grid: { stroke: 'var(--line-faint)' },
                    size: 50,
                    values: (_self, splits) => splits.map((v) => fmtValue(m, v))
                }
            ],
            series: [
                {},
                {
                    label: metricLabel(),
                    stroke: 'rgb(var(--ink))',
                    width: 1.5,
                    fill: 'rgba(var(--ink), 0.05)',
                    points: { show: false }
                }
            ]
        };
    }

    function rebuild(m: MetricId): void {
        if (!container) return;
        nowUnix = Math.floor(Date.now() / 1000);
        const { xs, ys } = series(m, completed);
        const data: uPlot.AlignedData = [xs.length ? xs : [nowUnix], ys.length ? ys : [0]];
        if (plot) {
            plot.destroy();
        }
        plot = new uPlot(buildOpts(container.clientWidth || 800, m), data, container);
    }

    onMount(() => {
        rebuild(metric);
        observer = new ResizeObserver(() => {
            if (plot) plot.setSize({ width: container.clientWidth, height: chartHeight });
        });
        observer.observe(container);
    });

    onDestroy(() => {
        observer?.disconnect();
        plot?.destroy();
    });

    $: if (plot && completed) {
        nowUnix = Math.floor(Date.now() / 1000);
        const { xs, ys } = series(metric, completed);
        if (xs.length) plot.setData([xs, ys]);
    }

    function pick(m: MetricId): void {
        if (m === metric) return;
        metric = m;
        rebuild(m);
    }
</script>

<section>
    <div class="flex items-center justify-between border-b dashed pb-1 mb-2">
        <span class="mono text-[11px] uppercase tracking-widest font-bold">Metrics</span>
        <div class="flex gap-1.5">
            {#each METRICS as m}
                <button
                    type="button"
                    on:click={() => pick(m.id)}
                    class={cls(
                        'border dashed px-[8px] py-[2px] mono text-[10px] uppercase tracking-widest hover:bg-ink hover:text-ink-inv',
                        m.id === metric && 'bg-ink text-ink-inv'
                    )}
                >
                    {m.label}
                </button>
            {/each}
        </div>
    </div>
    <div class="w-full" bind:this={container}></div>
    <div class="flex justify-between mono text-[10px] uppercase tracking-wider opacity-55 mt-1">
        <span>-30m</span>
        <span>now</span>
    </div>
</section>
