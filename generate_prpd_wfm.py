"""WFM adapter preserving the original generate_prpd.py processing and image style.
Only the input stage changes. All three channels must be present on the same grid.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert
from PIL import Image
from mso44_wfm import read_triplet

CLASS_MAPPING = {"xf": "xf_悬浮", "qx": "qx_气隙", "jswl": "jswl_金属微粒", "dy": "dy_电晕"}

def get_class_name(filename):
    """根据文件名自动识别类别"""
    filename_lower = filename.lower()
    # 按照 key 长度降序排序，防止匹配冲突
    for key in sorted(CLASS_MAPPING.keys(), key=len, reverse=True):
        if key in filename_lower:
            return CLASS_MAPPING[key]
    return "unknown_class"

def get_phase_from_ch2(t, ch2, fs):
    """从 CH2 提取工频相位 (使用过零点插值消除相位漂移)"""
    ch2_dc = ch2 - np.median(ch2)
    sos_lp = butter(2, 200/(fs/2), btype="low", output="sos")
    ch2_smooth = sosfiltfilt(sos_lp, ch2_dc)
    
    # Same crossing/interpolation calculation, vectorized for long WFM records.
    i = np.flatnonzero((ch2_smooth[:-1] < 0) & (ch2_smooth[1:] >= 0)) + 1
    frac = -ch2_smooth[i-1] / (ch2_smooth[i] - ch2_smooth[i-1])
    zc_times = t[i-1] + frac * (t[i] - t[i-1])
            
    if len(zc_times) < 2:
        return None, None, None
        
    zc_times = np.array(zc_times)
    periods = np.diff(zc_times)
    periods = periods[(periods > 0.015) & (periods < 0.025)] 
    f_grid = 1.0 / np.median(periods) if len(periods) > 0 else 50.0
    
    zc_phases = np.arange(len(zc_times)) * 360.0
    phase_all = np.interp(t, zc_times, zc_phases) % 360.0
    
    return phase_all, ch2_smooth, f_grid

def extract_pulses(ch, fs, hp_freq, thr_mult):
    """通用脉冲提取 (包络检波)"""
    ch_dc = ch - np.median(ch)
    sos_hp = butter(4, hp_freq/(fs/2), btype="highpass", output="sos")
    ch_hp = sosfiltfilt(sos_hp, ch_dc)
    
    env = np.abs(hilbert(ch_hp))
    sigma = 1.4826 * np.median(np.abs(env - np.median(env)))
    thr = thr_mult * sigma
    
    peaks, _ = find_peaks(env, height=thr, distance=int(fs*10e-6))
    return peaks, ch_hp

def draw_prpd_224(phase, amp, save_path, y_limit=None):
    """绘制 224x224 的无边框、无文字灰度图"""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    fig.patch.set_facecolor('black') 
    
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    
    if len(phase) > 0:
        pos_mask = amp > 0
        neg_mask = amp < 0
        
        # 正半周：白色；负半周：灰色
        ax.scatter(phase[pos_mask], amp[pos_mask], c='white', s=1.5, alpha=0.8, edgecolors='none')
        ax.scatter(phase[neg_mask], -amp[neg_mask], c='gray', s=1.5, alpha=0.8, edgecolors='none') 
        
        y_max = np.percentile(np.abs(amp), 99)
        if y_max == 0: y_max = 0.1 
        ax.set_ylim(0, y_max * 1.1)
    
    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    ax.set_xlim(0, 360)
    ax.set_axis_off()
    
    # 保存并去除白边
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
    
    # 强制转为 224x224 灰度图
    try:
        img = Image.open(save_path).convert('L')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img.save(save_path)
    except Exception as e:
        print(f"\n图像转换失败: {e}")

def process_record(ch1_path, output_root):
    ch1_path = Path(ch1_path)
    if not ch1_path.stem.lower().endswith('_ch1'):
        raise ValueError('Expected a *_ch1.wfm filename')
    stem = ch1_path.stem[:-4]
    paths = [ch1_path.with_name(stem + f'_ch{ch}.wfm') for ch in (1, 2, 3)]
    waves = read_triplet(*paths)
    if not all(w.verify_checksum() for w in waves.values()):
        raise ValueError('WFM checksum failed')
    fs = waves['uhf'].info.sample_rate_hz
    if fs <= 2_000_000:
        raise ValueError('Sample rate too low for original 1 MHz high-pass')
    class_name = get_class_name(stem)
    if class_name == 'unknown_class':
        raise ValueError('Unknown class prefix: ' + stem)
    targets = [Path(output_root)/role/class_name/(stem+'.png') for role in ('uhf','ic')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite existing pair: ' + str(targets))
    t = waves['phase_ref'].times()
    phase_all, _, frequency = get_phase_from_ch2(t, waves['phase_ref'].values(), fs)
    if phase_all is None:
        raise ValueError('Phase extraction failed: ' + stem)
    counts = {}
    for role, cutoff, target in zip(('uhf','hfct'), (1_000_000,500_000), targets):
        peaks, filtered = extract_pulses(waves[role].values(), fs, cutoff, 6.0)
        target.parent.mkdir(parents=True, exist_ok=True)
        draw_prpd_224(phase_all[peaks], filtered[peaks], str(target))
        with Image.open(target) as image:
            if image.size != (224,224) or image.mode != 'L':
                raise ValueError('Invalid output image')
        counts[role] = len(peaks)
    return dict(record_id=stem, class_name=class_name, source_files=[str(p) for p in paths],
                source_group=str(ch1_path.parent), images=[str(p) for p in targets],
                detected_peak_counts=counts, reference_frequency_hz=frequency,
                algorithm='Original generate_prpd.py functions; WFM input adapter',
                split=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ch1',nargs='+',type=Path,help='Explicit CH1 files; CH2/CH3 resolved by suffix')
    group.add_argument('--data-root',type=Path,help='Recursively find *_ch1.wfm under this root')
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    paths=args.ch1 if args.ch1 else sorted(p for p in args.data_root.rglob('*')
                                           if p.is_file() and p.name.lower().endswith('_ch1.wfm'))
    if not paths:
        raise ValueError('No CH1 files found')
    manifest=args.output_root/'manifest.json'
    if manifest.exists():
        raise FileExistsError('Choose a new output directory to preserve the existing manifest')
    # Preflight all pairs and avoid duplicate output filenames before processing.
    seen=set()
    for p in paths:
        stem=p.stem[:-4]
        key=(get_class_name(stem),stem)
        if key in seen:
            raise ValueError('Duplicate output record: '+stem)
        seen.add(key)
        for ch in (1,2,3):
            source=p.with_name(stem+f'_ch{ch}.wfm')
            if not source.is_file():
                raise FileNotFoundError(source)
    records=[]
    for p in paths:
        record=process_record(p,args.output_root)
        records.append(record)
        # Persist completed records even if a later record fails.
        manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(record,ensure_ascii=True))


if __name__ == '__main__':
    main()


# Legacy phase-tile functions recovered from 数据处理/cut_prpd.py.
# Source SHA256: 23e53f2cb3d8edcb515fd4210998b23c2134fc9d01c1371c688c048911ddf59b
# Only names are prefixed; numerical and rendering operations are preserved.

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
    N = ch2.shape[0]
    t = np.arange(N) / fs
    ref = ch2 - np.mean(ch2)
    ref = legacy_movmean(ref, smooth_win)
    zc = legacy_rising_zero_crossings(ref)
    if zc.size < 2:
        raise RuntimeError('CH2 上升零交叉少于 2 个，无法形成周期。')
    tZ = np.zeros(zc.size, dtype=float)
    eps = 1e-12
    for k, i1 in enumerate(zc):
        y1 = ref[i1]
        y2 = ref[i1 + 1]
        alpha = -y1 / (y2 - y1 + eps)
        tZ[k] = t[i1] + alpha * (t[i1 + 1] - t[i1])
    phi = np.full(N, np.nan, dtype=float)
    m = np.searchsorted(tZ, t, side='right') - 1
    valid = (m >= 0) & (m < len(tZ) - 1)
    m_valid = m[valid]
    T = tZ[m_valid + 1] - tZ[m_valid]
    phi_valid = 360.0 * (t[valid] - tZ[m_valid]) / (T + eps)
    phi_valid = np.mod(phi_valid, 360.0)
    phi[valid] = phi_valid
    return (phi, tZ)

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

def legacy_normalize_amplitude_global(amp: np.ndarray, a_ref: float, mode: str='asinh', gamma: float=2.0) -> np.ndarray:
    if amp.size == 0:
        return amp
    x = np.clip(amp / (a_ref + 1e-12), 0, None)
    if mode == 'linear':
        return np.clip(x, 0, 1.0)
    if mode == 'log1p':
        y = np.log1p(gamma * x) / np.log1p(gamma)
        return np.clip(y, 0, 1.0)
    y = np.arcsinh(gamma * x) / np.arcsinh(gamma)
    return np.clip(y, 0, 1.0)

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
