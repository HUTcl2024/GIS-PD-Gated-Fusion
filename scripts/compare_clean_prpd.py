"""Small clean-only PRPD comparison with identical hysteresis CH2 phase.

One deterministic WFM validation parent per class. Columns: signed sample
density, absolute sample density, absolute-amplitude detected peaks. Units are
microvolts; no asinh/clipping or image augmentation. No model/data rewrites.
Density counts are samples, not PD events. Phase and peak APIs are reused.
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
from plot_prpd_density import hysteresis_phase, density


def render(record_id, panels, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, role in enumerate(['uhf', 'hfct']):
        item = panels[role]
        for col, key in enumerate(['signed', 'absolute']):
            counts, pe, ae, _ = item[key]
            im = axes[row,col].pcolormesh(pe, ae, np.log1p(counts.T), cmap='Greys',
                shading='flat', vmin=0, rasterized=True)
            fig.colorbar(im, ax=axes[row,col], label='ln(1 + samples/bin)')
            axes[row,col].set_title(role.upper()+(' signed sample density' if col==0 else ' absolute sample density'))
        ph, amp = item['peaks']
        axes[row,2].scatter(ph, amp*1e6, s=1, alpha=.6, linewidths=0, rasterized=True)
        axes[row,2].set_title(f'{role.upper()} detected peaks (n={len(amp)})')
        top = item['absolute'][2][-1]
        # Absolute density and peaks use exactly the same amplitude limits.
        for col in [1,2]:
            axes[row,col].set_ylim(0,top)
            axes[row,col].axhline(item['threshold']*1e6, color='darkred', ls='--', lw=.7)
        for col in range(3):
            axes[row,col].set_xlim(0,360)
            axes[row,col].set_xticks([0,90,180,270,360])
            axes[row,col].set_xlabel('Phase (degrees)')
            axes[row,col].set_ylabel('Amplitude (microvolts)')
    fig.suptitle(record_id+' | same hysteresis phase, clean waveform\nNo amplitude normalization/clipping; red line = peak detection threshold')
    fig.savefig(output/f'{record_id}_comparison.png',dpi=300)
    plt.close(fig)


def main(args):
    if args.output.exists():
        raise FileExistsError('Choose a new comparison output directory')
    raw = (args.data/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest.get('representation') != 'legacy_phase_tiles_30deg':
        raise ValueError('Expected the archived tile manifest')
    parents = {}
    for r in manifest['records']:
        if r['split'] == 'val':
            parents.setdefault(r['parent_record_id'], dict(r,record_id=r['parent_record_id']))
    if args.record_ids:
        chosen = [parents[rid] for rid in args.record_ids]
        if len(set(args.record_ids)) != len(chosen):
            raise ValueError('Repeated parent ID')
    else:
        chosen = []
        for label in manifest['classes']:
            candidates = sorted((r for r in parents.values() if r['class_name']==label
                and r.get('source_format','wfm')=='wfm'),key=lambda r:r['record_id'])
            if not candidates:
                raise ValueError('No WFM validation parent for '+label)
            chosen.append(candidates[0])
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), split='val',
        selection='explicit IDs' if args.record_ids else 'first lexicographic WFM validation parent per class',
        source_code_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('plot_prpd_density.py'),ROOT/'generate_prpd_wfm.py']},
        note='Small diagnostic only. Does not establish best representation or independent PD counts.',records=[])
    for record in chosen:
        rid=record['record_id']
        print('Comparing:',record['class_name'],rid,flush=True)
        waves=read_parent_record(args.raw_data,record)
        phase_result=hysteresis_phase(waves['phase_ref'].values(),record['sample_rate_hz'])
        phase,crossings=phase_result[:2]  # Supports the server inline patch and repository API.
        periods=np.diff(crossings)
        if np.any((periods<.015)|(periods>.025)):
            raise ValueError('Invalid CH2 periods for '+rid+'; no interval deletion performed')
        entry=dict(record_id=rid,class_name=record['class_name'],cycles=len(periods),
            periods_ms=dict(min=float(periods.min()*1000),median=float(np.median(periods)*1000),max=float(periods.max()*1000)),channels={})
        panels={}
        for role in ['uhf','hfct']:
            signal=waves[role].values()
            signed=density(phase,signal)
            absolute=density(phase,np.abs(signal))
            ph,amp,threshold=legacy_detect_pulse_peaks(signal,phase,record['sample_rate_hz'],win_us=2.,thr_k=8.)
            panels[role]=dict(signed=signed,absolute=absolute,peaks=(ph,amp),threshold=threshold)
            entry['channels'][role]=dict(signed_density=signed[3],absolute_density=absolute[3],
                detected_peaks=len(amp),threshold_v=float(threshold),
                amplitude_peak_median_v=float(np.median(amp)) if len(amp) else None,
                amplitude_peak_max_v=float(amp.max()) if len(amp) else None)
            np.savez_compressed(args.output/f'{rid}_{role}_data.npz',
                signed_counts=signed[0],signed_amplitude_edges_uv=signed[2],
                absolute_counts=absolute[0],absolute_amplitude_edges_uv=absolute[2],
                phase_edges_deg=signed[1],peak_phase_deg=ph,peak_amplitude_v=amp)
            print(role,'samples=',signed[3]['counted_samples'],'peaks=',len(amp),'threshold V=',threshold,flush=True)
            del signal
        render(rid,panels,args.output)
        report['records'].append(entry)
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        del waves,phase,phase_result,panels
    print('Completed clean comparison:',args.output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record-ids',nargs='+')
    main(parser.parse_args())
