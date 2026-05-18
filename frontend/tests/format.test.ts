import { describe, expect, test } from 'vitest';
import {
    cls,
    fmtAgeLabel,
    fmtBytes,
    fmtCount,
    fmtDurationSec,
    fmtPoints,
    fmtTime,
    shortHotkey,
    shortWorker
} from '../src/lib/format';

describe('shortHotkey', () => {
    test('short strings pass through', () => {
        expect(shortHotkey('abc')).toBe('abc');
    });
    test('long strings get the ellipsis treatment', () => {
        expect(shortHotkey('5G98765432abcdef')).toBe('5G987\u2026cdef');
    });
    test('null becomes --', () => {
        expect(shortHotkey(null)).toBe('--');
    });
});

describe('shortWorker', () => {
    test('takes the part after the last hyphen', () => {
        expect(shortWorker('5G123-gpu0')).toBe('gpu0');
    });
    test('falls back to short hotkey for no hyphen', () => {
        expect(shortWorker('5G98765432abcdef')).toBe('5G987\u2026cdef');
    });
});

describe('fmtBytes', () => {
    test('bytes', () => expect(fmtBytes(512)).toBe('512 B'));
    test('KB', () => expect(fmtBytes(2048)).toBe('2.0 KB'));
    test('MB', () => expect(fmtBytes(5 * 1024 * 1024)).toBe('5.00 MB'));
    test('GB', () => expect(fmtBytes(2 * 1024 ** 3)).toBe('2.00 GB'));
});

describe('fmtDurationSec', () => {
    test('milliseconds', () => expect(fmtDurationSec(0.5)).toBe('500ms'));
    test('seconds', () => expect(fmtDurationSec(7.5)).toBe('7.5s'));
    test('minutes', () => expect(fmtDurationSec(125)).toBe('2m 5s'));
    test('hours', () => expect(fmtDurationSec(3700)).toBe('1h 1m'));
    test('days', () => expect(fmtDurationSec(86400 * 2 + 3600)).toBe('2d 1h'));
    test('negative / NaN', () => {
        expect(fmtDurationSec(-1)).toBe('--');
        expect(fmtDurationSec(NaN)).toBe('--');
    });
});

describe('fmtAgeLabel', () => {
    test('appends ago', () => expect(fmtAgeLabel(15)).toBe('15s ago'));
    test('null', () => expect(fmtAgeLabel(null)).toBe('--'));
});

describe('fmtTime', () => {
    test('null', () => expect(fmtTime(null)).toBe('--'));
    test('produces a non-empty string', () => {
        expect(fmtTime(1700000000)).not.toBe('--');
    });
});

describe('fmtPoints', () => {
    test('small numbers', () => expect(fmtPoints(0.05)).toBe('0.05'));
    test('hundreds', () => expect(fmtPoints(123)).toBe('123'));
    test('thousands', () => expect(fmtPoints(5500)).toBe('5.5k'));
});

describe('fmtCount', () => {
    test('formats with thousand separators', () => {
        const v = fmtCount(1234567);
        expect(v.replace(/[\s\u00A0]/g, '')).toMatch(/1[.,]234[.,]567/);
    });
});

describe('cls', () => {
    test('joins truthy parts', () => {
        expect(cls('a', false, 'b', null, 'c')).toBe('a b c');
    });
});
