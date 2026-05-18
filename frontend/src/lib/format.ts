/**
 * Formatting helpers shared by every panel.
 *
 * Pure functions only; no DOM, no Svelte, easy to unit test under jsdom.
 */

export function shortHotkey(value: string | null | undefined, head = 5, tail = 4): string {
    if (!value) return '--';
    if (value.length <= head + tail + 3) return value;
    return `${value.slice(0, head)}\u2026${value.slice(-tail)}`;
}

export function shortWorker(value: string | null | undefined): string {
    if (!value) return '--';
    if (value.includes('-')) return value.split('-').pop() ?? value;
    return shortHotkey(value);
}

export function fmtBytes(n: number | null | undefined): string {
    const v = Number(n || 0);
    if (v < 1024) return `${v} B`;
    if (v < 1024 ** 2) return `${(v / 1024).toFixed(1)} KB`;
    if (v < 1024 ** 3) return `${(v / 1024 ** 2).toFixed(2)} MB`;
    return `${(v / 1024 ** 3).toFixed(2)} GB`;
}

export function fmtDurationSec(n: number | null | undefined): string {
    const v = Number(n);
    if (!Number.isFinite(v) || v < 0) return '--';
    if (v < 1) return `${Math.round(v * 1000)}ms`;
    if (v < 60) return `${v < 10 ? v.toFixed(1) : Math.round(v)}s`;
    let total = Math.round(v);
    const days = Math.floor(total / 86400);
    total -= days * 86400;
    const hours = Math.floor(total / 3600);
    total -= hours * 3600;
    const minutes = Math.floor(total / 60);
    const seconds = total - minutes * 60;
    if (days > 0) return `${days}d${hours > 0 ? ` ${hours}h` : ''}`;
    if (hours > 0) return `${hours}h${minutes > 0 ? ` ${minutes}m` : ''}`;
    return `${minutes}m${seconds > 0 ? ` ${seconds}s` : ''}`;
}

export function fmtAgeLabel(seconds: number | null | undefined): string {
    if (seconds == null) return '--';
    return `${fmtDurationSec(seconds)} ago`;
}

export function fmtTime(unix: number | null | undefined): string {
    if (!unix) return '--';
    return new Date(unix * 1000).toLocaleTimeString();
}

export function fmtPoints(n: number | null | undefined): string {
    const v = Number(n || 0);
    if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
    if (v >= 100) return v.toFixed(0);
    return v.toFixed(2);
}

export function fmtCount(n: number | null | undefined): string {
    return new Intl.NumberFormat().format(Number(n || 0));
}

/** Compose a comma-joined CSS class list, dropping falsy entries. */
export function cls(...parts: (string | false | null | undefined)[]): string {
    return parts.filter(Boolean).join(' ');
}
