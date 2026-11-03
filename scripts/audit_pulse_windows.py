"""Inspect clean validation detections in raw time-domain windows; no training.

Uses corrected hysteresis CH2 phase and EXACT legacy peak grouping. Does not
label peaks as physical PD, change thresholds, or rewrite model input images.
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
from revision_data import read_parent_record
from generate_prpd_wfm import legacy_detect_pulse_peaks
from plot_prpd_density import hysteresis_phase


def indexed_peaks(signal, phase, fs):
    """Recover original sample indices without modifying legacy semantics."""
    valid = np.flatnonzero(~np.isnan(phase))
    amplitude = np.abs(signal[valid])
    median = np.median(amplitude)
    threshold = median + 8 * (np.median(np.abs(amplitude - median)) + 1e-12)
    above = np.flatnonzero(amplitude > threshold)
    window = max(5, int(round(2e-6 * fs)))
    groups = np.split(above, np.flatnonzero(np.diff(above) > window) + 1)
    groups = [g for g in groups if len(g)]
    peaks = np.asarray([valid[g[np.argmax(amplitude[g])]] for g in groups], dtype=int)
    widths = np.asarray([(valid[g[-1]] - valid[g[0]]) / fs * 1e6 for g in groups])
    return peaks, float(threshold), widths


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in [0, .1, .5, .9, 1]} if len(values) else None


def anchors(signal, peaks):
    if not len(peaks):
        return []
    order = np.argsort(np.abs(signal[peaks]), kind='stable')
    chosen = [('near threshold (10th percentile)', int(peaks[order[int(.1*(len(order)-1))]])),
              ('median detected amplitude', int(peaks[order[(len(order)-1)//2]])),
              ('maximum detected amplitude', int(peaks[order[-1]]))]
    if len(peaks) > 1:
        chosen.append(('closest detected pair: first peak', int(peaks[np.argmin(np.diff(peaks))])))
    return chosen


def draw_window(ax, signal, peaks, center, fs, threshold, width_us):
    half = max(1, round(width_us * 1e-6 * fs / 2))
    lo, hi = max(0, center-half), min(len(signal), center+half+1)
    indices = np.arange(lo, hi)
    ax.plot((indices-center)/fs*1e6, signal[lo:hi]*1e3, lw=.65)
    selected = peaks[(peaks >= lo) & (peaks < hi)]
    ax.scatter((selected-center)/fs*1e6, signal[selected]*1e3, s=10, c='crimson', zorder=3)
    for value in [-threshold, threshold]:
        ax.axhline(value*1e3, color='darkred', ls='--', lw=.5)
    ax.axvline(0, color='gray', lw=.4)


def main(args):
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; nothing overwritten')
    raw = (args.data/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest.get('representation') != 'legacy_phase_tiles_30deg':
        raise ValueError('Expected 12-tile manifest')
    parents = {}
    for record in manifest['records']:
        if record['split'] == 'val':
            rid = record['parent_record_id']
            parents.setdefault(rid, dict(record, record_id=rid))
    chosen = [parents[rid] for rid in args.record_ids]  # fails before creating output on missing IDs
    if len(set(args.record_ids)) != len(chosen):
        raise ValueError('Repeated parent IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), split='val',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('plot_prpd_density.py'), ROOT/'generate_prpd_wfm.py']},
        note='Clean waveform diagnostic. Threshold/grouping unchanged. Counts are detections, not verified PD events. '
             'Closest-pair selection is intentionally extreme, not representative.', records=[])
    for record in chosen:
        rid, fs = record['record_id'], float(record['sample_rate_hz'])
        print('Reading clean validation record:', rid, flush=True)
        waves = read_parent_record(args.raw_data, record)
        phase, crossings = hysteresis_phase(waves['phase_ref'].values(), fs)[:2]
        periods = np.diff(crossings)
        if np.any((periods < .015) | (periods > .025)):
            raise ValueError('Invalid CH2 period; stop without discarding intervals: '+rid)
        entry = dict(record_id=rid, class_name=record['class_name'],
                     complete_cycles=len(periods), period_ms=quantiles(periods*1000), channels={})
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
        for row, role in enumerate(['uhf', 'hfct']):
            signal = waves[role].values()
            peaks, threshold, widths = indexed_peaks(signal, phase, fs)
            ph, amp, original_threshold = legacy_detect_pulse_peaks(signal, phase, fs, win_us=2., thr_k=8.)
            if not (np.array_equal(ph, phase[peaks]) and np.array_equal(amp, np.abs(signal[peaks]))
                    and original_threshold == threshold):
                raise ValueError('Recovered indices differ from legacy detector')
            selected = anchors(signal, peaks)
            gaps = np.diff(peaks)/fs*1e6
            item = dict(detections=len(peaks), threshold_v=threshold,
                        detection_gap_us=quantiles(gaps), group_span_us=quantiles(widths),
                        gaps_below_10us=int(np.sum(gaps < 10)),
                        waveform_median_v=float(np.median(signal)),
                        amplitude_v=quantiles(amp), inspected=[])
            for col, ax in enumerate(axes[row]):
                if col >= len(selected):
                    ax.set_axis_off()
                    continue
                label, center = selected[col]
                draw_window(ax, signal, peaks, center, fs, threshold, args.window_us)
                inset = ax.inset_axes([.55, .63, .43, .34])
                draw_window(inset, signal, peaks, center, fs, threshold, args.zoom_us)
                inset.tick_params(labelsize=5)
                inset.set_title(f'{args.zoom_us:g} us zoom', fontsize=6)
                ax.set_title(f'{role.upper()} | {label}\nphase={phase[center]:.2f} deg', fontsize=9)
                ax.set_xlabel('Time relative to selected peak (us)')
                ax.set_ylabel('Raw signed voltage (mV)')
                item['inspected'].append(dict(selection=label, sample_index=center,
                    time_s=center/fs, phase_deg=float(phase[center]), signed_voltage_v=float(signal[center])))
            entry['channels'][role] = item
            print(f'{role}: detections={len(peaks)}, threshold={threshold:.6g} V, '
                  f'min gap={float(gaps.min()) if len(gaps) else None} us', flush=True)
        fig.suptitle(rid+' | clean raw windows, corrected CH2 phase\n'
                     'Red dots: legacy detections; dashed lines: +/- threshold; insets: local zoom. No filtering.')
        fig.savefig(args.output/f'{rid}_pulse_windows.png', dpi=180)
        plt.close(fig)
        report['records'].append(entry)
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        del signal, phase, waves
    print('Done: four diagnostic figures and report.json; no model/input image changes.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--raw-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record-ids', nargs='+', default=[
        'jd-6.9kv3_000', 'qx-10kv101_000', 'xf-5.3kv1', 'ym-12.4kv12_000'])
    parser.add_argument('--window-us', type=float, default=100.)
    parser.add_argument('--zoom-us', type=float, default=10.)
    args = parser.parse_args()
    if not (0 < args.zoom_us <= args.window_us):
        parser.error('Require 0 < zoom-us <= window-us')
    main(args)