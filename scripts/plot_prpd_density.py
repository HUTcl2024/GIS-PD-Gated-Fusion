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
from generate_prpd_wfm import legacy_compute_phase_from_ch2


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
    phase, crossings = legacy_compute_phase_from_ch2(waves['phase_ref'].values(), record['sample_rate_hz'], smooth_win=21)
    periods = np.diff(crossings)
    if not len(periods) or np.any((periods < .015) | (periods > .025)):
        raise ValueError('CH2 has intervals outside 15-25 ms; inspect phase reference before plotting')
    results = {}
    report = dict(record_id=args.record_id, split='val', class_name=record['class_name'],
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        phase_bins=256, amplitude_bins=128, phase_source='CH2 rising-zero crossings, 21-point smoothing',
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
    main(parser.parse_args())
