"""Shared WFM/CSV input and raw-waveform AWGN helpers for full360 PRPD."""
import hashlib
import json
from pathlib import Path
import numpy as np
from mso44_wfm import read_triplet
ROOT=Path(__file__).resolve().parent

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def image_paths(out, scenario, record):
    return [out/'images'/scenario/role/record['class_name']/(record['record_id']+'.png')
            for role in ['uhf','hfct']]

def read_parent_record(root,record):
    if record.get('source_format','wfm')=='csv':
        from mso44_csv import CsvRecord
        source=CsvRecord(root/record['sources'][0])
        if source.sha256!=record['source_sha256']: raise ValueError('CSV changed since parent manifest')
        return source.waves
    waves=read_triplet(*(root/p for p in record['sources']))
    if not all(w.verify_checksum() for w in waves.values()): raise ValueError('WFM checksum failed')
    return waves

def add_awgn_snr(signal,snr_db,rng):
    """Measured SNR relative to the complete median-centered captured waveform."""
    clean=np.asarray(signal,dtype=np.float64)
    power=float(np.mean((clean-np.median(clean))**2))
    if not np.isfinite(power) or power<=0: raise ValueError('SNR requires positive finite waveform power')
    noise=rng.standard_normal(len(clean)); noise-=noise.mean()
    noise*=np.sqrt(power*10.**(-snr_db/10.)/np.mean(noise**2))
    actual=10.*np.log10(power/np.mean(noise**2))
    return clean+noise,dict(requested_snr_db=snr_db,measured_snr_db=float(actual),
        signal_power_v2=power,noise_power_v2=float(np.mean(noise**2)))

def legacy_pulses(root,record,fraction=0.,target='both',snr_db=None,noise_quality=None,noise_seed=2026):
    from generate_prpd_wfm import legacy_compute_phase_from_ch2,legacy_detect_pulse_peaks
    if fraction: raise ValueError('Only fixed-SNR AWGN is supported; no percentage/composite corruption')
    waves=read_parent_record(root,record)
    fs=record['sample_rate_hz']
    phase,_=legacy_compute_phase_from_ch2(waves['phase_ref'].values(),fs,smooth_win=21)
    result={}
    for role in ['uhf','hfct']:
        signal=waves[role].values()
        if snr_db is not None and target in [role,'both']:
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{noise_seed}/awgn/{record["record_id"]}/{role}'.encode()).digest()[:8],'little'))
            signal,quality=add_awgn_snr(signal,snr_db,rng)
            if noise_quality is not None: noise_quality[role]=quality
        result[role]=legacy_detect_pulse_peaks(signal,phase,fs,win_us=2.,thr_k=8.)[:2]
    return result
