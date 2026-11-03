"""Single clean-record phase/amplitude sample-density diagnostic.

256 phase bins x 128 signed amplitude bins; all complete CH2-referenced cycles
are accumulated. Counts are waveform samples, NOT independent PD events.
No peak selection, filtering, baseline subtraction, asinh mapping or clipping.
Amplitude limits come from this one record for visualization only, not a fitted
training normalization. Does not create/replace model training images.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
from PIL import Image
from revision_data import read_parent_record
from generate_prpd_wfm import legacy_compute_phase_from_ch2, legacy_movmean


def hysteresis_phase(ch2, fs):
    """Arm below -h, accept above +h; interpolate a rising zero in between.

    h=10% of the 99th percentile absolute centered/smoothed CH2. The threshold
    only identifies a complete transition; phase zero remains at zero volts.
    No 20ms rescheduling, interval deletion or fixed-frequency assumption.
    """
    ref = legacy_movmean(ch2-ch2.mean(), 21)
    h = .1*float(np.quantile(np.abs(ref), .99))
    if not np.isfinite(h) or h <= 0:
        raise ValueError('CH2 lacks a usable reference amplitude')
    below = np.where((ref[:-1] >= -h) & (ref[1:] < -h))[0]+1
    above = np.where((ref[:-1] < h) & (ref[1:] >= h))[0]+1
    zeros = np.where((ref[:-1] <= 0) & (ref[1:] > 0))[0]
    events = sorted([(int(i), False) for i in below]+[(int(i), True) for i in above])
    armed = 0 if ref[0] < -h else None
    times = []
    for i, positive in events:
        if not positive:
            armed = i
        elif armed is not None:
            left, right = np.searchsorted(zeros, [armed, i])
            candidates = zeros[left:right]
            if not len(candidates):
                raise ValueError('No rising zero between hysteresis thresholds')
            estimate = armed-ref[armed]*(i-armed)/(ref[i]-ref[armed])
            j = int(candidates[np.argmin(np.abs(candidates-estimate))])
            times.append((j-ref[j]/(ref[j+1]-ref[j]))/fs)
            armed = None
    crossings = np.asarray(times)
    if len(crossings) < 2:
        raise ValueError('Fewer than two complete hysteresis rising crossings')
    t = np.arange(len(ref))/fs
    cycle = np.searchsorted(crossings, t, side='right')-1
    valid = (cycle >= 0) & (cycle < len(crossings)-1)
    phase = np.full(len(ref), np.nan)
    indices = cycle[valid]
    phase[valid] = 360*(t[valid]-crossings[indices])/(crossings[indices+1]-crossings[indices])
    return phase, crossings, h


def density(phase, signal):
    valid = np.isfinite(phase) & np.isfinite(signal)
    x, y = phase[valid], signal[valid] * 1e6
    if not len(x):
        raise ValueError('No finite phase-aligned samples')
    low, high = float(y.min()), float(y.max())
    if low == high:
        low -= max(abs(low)*1e-6, 1e-6)
        high += max(abs(high)*1e-6, 1e-6)
    phase_edges = np.linspace(0., 360., 257)
    amp_edges = np.linspace(low, high, 129)
    counts, _, _ = np.histogram2d(x, y, bins=[phase_edges, amp_edges])
    counts = counts.astype(np.int64)
    if int(counts.sum()) != len(x):
        raise ValueError('Histogram lost samples')
    return counts, phase_edges, amp_edges, dict(total_samples=len(signal),
        counted_samples=len(x), excluded_samples=len(signal)-len(x),
        amplitude_min_uv=low, amplitude_max_uv=high,
        occupied_bins=int(np.count_nonzero(counts)), maximum_bin_count=int(counts.max()))


def draw(results, record_id, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    for row, role in enumerate(['uhf', 'hfct']):
        counts, phase_edges, amp_edges, info = results[role]
        for col, log in enumerate([False, True]):
            values = np.log1p(counts) if log else counts
            image = axes[row, col].pcolormesh(phase_edges, amp_edges, values.T,
                cmap='Greys', shading='flat', vmin=0, rasterized=True)
            axes[row, col].set(xlim=(0,360), xlabel='Phase (degrees)',
                ylabel='Signed amplitude (microvolts)',
                title=f'{role.upper()} | '+('log counts' if log else 'linear counts'))
            axes[row, col].set_xticks([0,90,180,270,360])
            fig.colorbar(image, ax=axes[row,col], label='ln(1 + samples/bin)' if log else 'Samples/bin')
        gray = np.log1p(counts.T)
        gray = 255*(1-gray/gray.max()) if gray.max() else np.full_like(gray, 255.)
        # 128 amplitude rows; flip to put positive/high amplitudes at image top.
        Image.fromarray(np.flipud(np.rint(gray).astype(np.uint8))).save(output/f'density_{role}.png')
    fig.suptitle(f'{record_id} | clean sample density\n256 phase x 128 amplitude bins')
    fig.savefig(output/'density_comparison.png', dpi=300)
    fig.savefig(output/'density_comparison.svg')
    fig.savefig(output/'density_comparison.pdf')
    plt.close(fig)


def phase_audit(ch2, fs, crossings, record_id, output):
    """Inspect the existing zero-crossing estimator, without fixing/excluding cycles."""
    intervals = np.diff(crossings)
    good = (intervals >= .015) & (intervals <= .025)
    stats = dict(record_id=record_id, samples=len(ch2), sample_rate_hz=fs,
        duration_s=len(ch2)/fs, rising_zero_crossings=len(crossings),
        interval_count=len(intervals), intervals_15_to_25_ms=int(good.sum()),
        intervals_below_15_ms=int((intervals < .015).sum()),
        intervals_above_25_ms=int((intervals > .025).sum()),
        interval_ms_quantiles={str(q): float(np.quantile(intervals, q)*1000)
                               for q in [0, .01, .1, .5, .9, .99, 1]} if len(intervals) else {},
        first_30_crossings_ms=(crossings[:30]*1000).tolist(),
        first_30_intervals_ms=(intervals[:30]*1000).tolist(),
        ch2_min_v=float(ch2.min()), ch2_max_v=float(ch2.max()),
        expected_50hz_period_ms=20., smoothing_samples=21,
        decision='Diagnostic only; no rejected intervals silently removed and no density generated')
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7), constrained_layout=True)
    def trace(ax, signal, stop):
        edges = np.linspace(0, stop, min(4000, stop)+1, dtype=int)
        low, high = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            low.append(float(signal[a:b].min())); high.append(float(signal[a:b].max()))
        t = (edges[:-1]+edges[1:])*.5/fs*1000
        ax.fill_between(t, low, high, alpha=.5, color='steelblue')
        ax.set_xlabel('Time from record start (ms)'); ax.set_ylabel('CH2 (V)')
    trace(axes[0], ch2, len(ch2))
    axes[0].set_title('Original CH2: full-record min/max envelope')
    centered = legacy_movmean(ch2-ch2.mean(), 21)
    stop = min(len(ch2), int(round(.06*fs)))
    trace(axes[1], centered, stop)
    selected = crossings[crossings < stop/fs]
    displayed = selected[np.linspace(0,len(selected)-1,min(4000,len(selected)),dtype=int)] if len(selected) else selected
    axes[1].scatter(displayed*1000, np.zeros(len(displayed)), s=5, color='darkred', label='detected rising crossings')
    axes[1].axhline(0, color='gray', lw=.5)
    axes[1].set_title('Mean-centered, 21-point smoothed CH2: first 60 ms')
    axes[1].legend()
    positive = intervals[intervals > 0]*1000
    if len(positive):
        low, high = float(positive.min()), float(positive.max())
        if high > low:
            axes[2].hist(positive, bins=np.geomspace(low, high, 60), color='steelblue')
            axes[2].set_xscale('log')
        else:
            axes[2].hist(positive, bins=10, color='steelblue')
    axes[2].axvline(20, color='darkred', ls='--', label='20 ms reference')
    axes[2].set(xlabel='Consecutive rising-crossing interval (ms)', ylabel='Count',
                title='All crossing intervals; no interval rejection')
    axes[2].legend()
    fig.suptitle(record_id+' | phase reference audit')
    fig.savefig(output/'phase_reference_audit.png', dpi=300)
    plt.close(fig)
    (output/'phase_reference_report.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)
    print('Phase reference audit saved. Density NOT generated:', output, flush=True)


def main(args):
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    raw = (args.data/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    records = [r for r in manifest['records'] if r.get('parent_record_id', r['record_id']) == args.record_id]
    if not records:
        raise ValueError('Record not found')
    if len({r['split'] for r in records}) != 1 or records[0]['split'] != 'val':
        raise ValueError('This diagnostic requires a validation record')
    record = dict(records[0], record_id=args.record_id)
    print('Reading one clean validation record:', args.record_id, flush=True)
    waves = read_parent_record(args.raw_data, record)
    ch2 = waves['phase_ref'].values()
    threshold = None
    if args.phase_method == 'hysteresis':
        phase, crossings, threshold = hysteresis_phase(ch2, record['sample_rate_hz'])
    else:
        phase, crossings = legacy_compute_phase_from_ch2(ch2, record['sample_rate_hz'], smooth_win=21)
    periods = np.diff(crossings)
    print('Phase method:', args.phase_method, '| crossings:', len(crossings),
          '| period ms min/median/max:', (np.array([periods.min(), np.median(periods), periods.max()])*1000).tolist(),
          '| hysteresis threshold V:', threshold, flush=True)
    if args.phase_audit_only or not len(periods) or np.any((periods < .015) | (periods > .025)):
        phase_audit(ch2, record['sample_rate_hz'], crossings, args.record_id, args.output)
        return
    del ch2
    results = {}
    report = dict(record_id=args.record_id, split='val', class_name=record['class_name'],
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        phase_bins=256, amplitude_bins=128, phase_source='CH2 rising-zero crossings, 21-point smoothing',
        phase_method=args.phase_method, hysteresis_threshold_v=threshold,
        cycles=int(len(periods)), period_ms=dict(min=float(periods.min()*1000),
            median=float(np.median(periods)*1000), max=float(periods.max()*1000)),
        cycle_selection='all complete CH2 cycles; incomplete leading/trailing samples excluded',
        amplitude='signed physical voltage x1e6, no baseline subtraction or amplitude clipping',
        amplitude_range='full finite min/max per sensor in this record; visualization only',
        display='linear counts and log1p counts; individual sensor color scales',
        count_definition='waveform samples per bin, not discharge-event counts', channels={})
    for role in ['uhf','hfct']:
        wave = waves[role]
        info = getattr(wave, 'info', None)
        if info is not None and getattr(info, 'y_unit', 'V') != 'V':
            raise ValueError('Expected voltage units before converting to microvolts')
        results[role] = density(phase, wave.values())
        report['channels'][role] = results[role][3]
        print(role, report['channels'][role], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    for role, (counts, pe, ae, _) in results.items():
        np.savez_compressed(args.output/f'counts_{role}.npz', counts=counts,
                            phase_edges_deg=pe, amplitude_edges_uv=ae)
    draw(results, args.record_id, args.output)
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--raw-data', type=Path, required=True)
    parser.add_argument('--record-id', default='jd-6.9kv3_000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase-audit-only', action='store_true', help='Inspect CH2 and crossing intervals; no density generation')
    parser.add_argument('--phase-method', choices=['legacy','hysteresis'], default='legacy')
    main(parser.parse_args())
