"""Corrected CH2 phase, original peak detection and full-PRPD scatter renderer.

The legacy_* API names remain for checkpoint/workflow compatibility. The active
pipeline uses complete360 raw-voltage images; no amplitude mapping is provided.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def legacy_rising_zero_crossings(ref: np.ndarray):
    return np.where((ref[:-1] <= 0) & (ref[1:] > 0))[0]

def legacy_movmean(x: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return x
    win = int(win)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(xp, kernel, mode='valid')

def legacy_compute_phase_from_ch2(ch2: np.ndarray, fs: float, smooth_win=21):
    """Hysteresis rising-zero phase; same API as the archived phase function.

    Arm below -h and accept above +h, h=10% of the 99th percentile absolute
    centered/smoothed CH2. Interpolate a rising ZERO between thresholds.
    Use measured consecutive periods; no forced 20ms grid or dropped periods.
    Reject any period outside 15-25ms for this 50Hz acquisition protocol.
    Leading/trailing incomplete cycles remain NaN as before.
    """
    ch2 = np.asarray(ch2, dtype=np.float64)
    if ch2.ndim != 1 or len(ch2) < 2 or not np.all(np.isfinite(ch2)):
        raise ValueError('CH2 must be a finite one-dimensional reference')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling rate must be positive and finite')
    if smooth_win is not None and (int(smooth_win) != smooth_win or smooth_win < 1 or int(smooth_win) % 2 != 1):
        raise ValueError('CH2 smoothing window must be a positive odd integer')
    ref = legacy_movmean(ch2 - np.mean(ch2), smooth_win)
    h = .1 * float(np.quantile(np.abs(ref), .99))
    if not np.isfinite(h) or h <= 0:
        raise ValueError('CH2 lacks a usable reference amplitude')
    below = np.where((ref[:-1] >= -h) & (ref[1:] < -h))[0] + 1
    above = np.where((ref[:-1] < h) & (ref[1:] >= h))[0] + 1
    zeros = legacy_rising_zero_crossings(ref)
    events = sorted([(int(i), False) for i in below] + [(int(i), True) for i in above])
    armed = 0 if ref[0] < -h else None
    times = []
    for i, positive in events:
        if not positive:
            armed = i
        elif armed is not None:
            left, right = np.searchsorted(zeros, [armed, i])
            candidates = zeros[left:right]
            if not len(candidates):
                raise ValueError('No rising zero between CH2 hysteresis thresholds')
            estimate = armed - ref[armed] * (i-armed) / (ref[i]-ref[armed])
            j = int(candidates[np.argmin(np.abs(candidates-estimate))])
            times.append((j-ref[j]/(ref[j+1]-ref[j])) / fs)
            armed = None
    crossings = np.asarray(times, dtype=np.float64)
    if len(crossings) < 2:
        raise ValueError('Fewer than two hysteresis rising crossings')
    periods = np.diff(crossings)
    if np.any((periods < .015) | (periods > .025)):
        raise ValueError(f'CH2 period outside 15-25ms: min={periods.min()*1000:.6f}, '
                         f'max={periods.max()*1000:.6f}; no intervals discarded')
    t = np.arange(len(ref)) / fs
    cycle = np.searchsorted(crossings, t, side='right') - 1
    valid = (cycle >= 0) & (cycle < len(crossings)-1)
    phase = np.full(len(ref), np.nan)
    indices = cycle[valid]
    phase[valid] = 360 * (t[valid]-crossings[indices]) / periods[indices]
    return phase, crossings

def legacy_detect_pulse_peaks(uhf: np.ndarray, phi: np.ndarray, fs: float, win_us=2.0, thr_k=8.0):
    valid = ~np.isnan(phi)
    uhf_v = uhf[valid]
    phi_v = phi[valid]
    uhf_abs = np.abs(uhf_v)
    medv = np.median(uhf_abs)
    madv = np.median(np.abs(uhf_abs - medv)) + 1e-12
    thr = medv + thr_k * madv
    win = max(5, int(round(win_us * 1e-06 * fs)))
    idx = np.where(uhf_abs > thr)[0]
    if idx.size == 0:
        return (np.array([]), np.array([]), thr)
    cut_points = np.where(np.diff(idx) > win)[0]
    starts = np.r_[0, cut_points + 1]
    ends = np.r_[cut_points, idx.size - 1]
    peaks = []
    for s, e in zip(starts, ends):
        seg = idx[s:e + 1]
        peak_i = seg[np.argmax(uhf_abs[seg])]
        peaks.append(peak_i)
    peaks = np.array(peaks, dtype=int)
    fai = phi_v[peaks]
    amp = uhf_abs[peaks]
    return (fai, amp, thr)

def legacy_save_scatter_tile(out_path: Path, fai_tile: np.ndarray, amp_tile: np.ndarray, xlim=(0.0, 30.0), ylim=(0.0, 1.0), s=8, alpha=0.9, dpi=256, bg_white=True, resize_to=224):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.56, 2.56), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    if bg_white:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    else:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    if fai_tile.size > 0:
        ax.scatter(fai_tile, amp_tile, s=s, marker='.', linewidths=0, alpha=alpha)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    if resize_to is not None and int(resize_to) > 0:
        resize_to = int(resize_to)
        with Image.open(out_path) as im:
            im2 = im.resize((resize_to, resize_to), resample=Image.Resampling.LANCZOS)
            im2.save(out_path)
