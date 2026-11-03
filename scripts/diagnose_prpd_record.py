"""One validation record: waveform -> pulses -> PRPD -> existing 12 tiles.

Recreates the fixed both-channel validation AWGN, seed2026. No training or
changes to source images/waveforms. Writes two diagnostic PNGs and one JSON.
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
from PIL import Image, ImageDraw
from revision_data import read_parent_record, add_awgn_snr, image_paths
from generate_prpd_wfm import legacy_compute_phase_from_ch2, legacy_detect_pulse_peaks, legacy_normalize_amplitude_global


def envelope(x, fs, first, last, bins=1000):
    edges = np.linspace(first, last, min(bins, last-first)+1, dtype=int)
    t, low, high = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            t.append(1000*((a+b)/2-first)/fs)
            low.append(float(x[a:b].min()))
            high.append(float(x[a:b].max()))
    return np.array(t), np.array(low), np.array(high)


def main(args):
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('representation') != 'legacy_phase_tiles_30deg':
        raise ValueError('Expected 12-tile data')
    candidates = {}
    for r in manifest['records']:
        if r['split'] == 'val':
            candidates.setdefault(r['parent_record_id'], []).append(r)
    if args.record_id:
        parent_id = args.record_id
    else:
        ids = sorted(k for k, v in candidates.items() if v[0]['class_name'] == args.class_name
                     and v[0].get('source_format', 'wfm') == 'wfm')
        if not ids:
            raise ValueError('No WFM validation record in requested class; specify --record-id')
        parent_id = ids[0]
    tiles = sorted(candidates[parent_id], key=lambda r: r['phase_start_deg'])
    if len(tiles) != 12:
        raise ValueError('Expected twelve tiles')
    parent = dict(tiles[0], record_id=parent_id)
    scenarios = ['clean']+[f'val_awgn_both_snr{db}dB' for db in [-15, -10, -5]]
    for scenario in scenarios:
        for tile in tiles:
            for p in image_paths(args.data, scenario, tile):
                if not p.is_file():
                    raise FileNotFoundError(p)
    print('Selected validation record:', parent_id, flush=True)
    waves = read_parent_record(args.raw_data, parent)
    fs = parent['sample_rate_hz']
    phase, crossings = legacy_compute_phase_from_ch2(waves['phase_ref'].values(), fs, smooth_win=21)
    periods = [(a, b) for a, b in zip(crossings[:-1], crossings[1:]) if .015 <= b-a <= .025]
    if not periods:
        raise ValueError('No complete 15-25ms reference cycle')
    a, b = periods[0]
    first, last = int(np.ceil(a*fs)), int(np.ceil(b*fs))
    refs = manifest['preprocessing']['a_ref']
    values = {}
    report = dict(record_id=parent_id, class_name=parent['class_name'], split='val',
        manifest_sha256=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        sample_rate_hz=fs, references=refs, waveform_cycle_seconds=[a,b],
        waveform_plot='min/max envelope of the same first complete cycle in all conditions',
        prpd_plot='all valid cycles; at most 20000 evenly indexed pulses displayed, statistics use all pulses',
        note='Count ratios are not matched-event retention. A single record cannot diagnose the entire dataset.',
        conditions={})
    for role in ['uhf', 'hfct']:
        clean = waves[role].values()
        for scenario, db in zip(scenarios, [None, -15, -10, -5]):
            quality = {}
            if db is None:
                signal = clean
            else:
                seed = int.from_bytes(hashlib.sha256(f'2026/awgn/{parent_id}/{role}'.encode()).digest()[:8], 'little')
                signal, quality = add_awgn_snr(clean, db, np.random.default_rng(seed))
            ph, amp, threshold = legacy_detect_pulse_peaks(signal, phase, fs, win_us=2., thr_k=8.)
            mapped = legacy_normalize_amplitude_global(amp, refs[role], mode='asinh', gamma=2.)
            counts = [int(((ph >= x)&(ph < x+30)).sum()) for x in range(0,360,30)]
            # Compare recreated pulse counts against the original rendering audit.
            quality_path = args.data/'images'/scenario/'quality'/(parent_id+'.json')
            if quality_path.is_file():
                old = json.loads(quality_path.read_text(encoding='utf-8'))
                expected = [old['tile_peak_counts'][t['record_id']][role] for t in tiles]
                if counts != expected:
                    raise ValueError(f'Recreated pulse counts differ from saved images: {scenario}/{role}')
            baseline = len(values['clean',role]['amp']) if db is not None else len(amp)
            row = dict(pulses=len(amp), threshold_v=float(threshold),
                pulse_count_ratio_pct=100*len(amp)/baseline if baseline else None,
                clipped_pct=100*float(np.mean(amp >= refs[role])) if len(amp) else 0.,
                amplitude_median_v=float(np.median(amp)) if len(amp) else None,
                amplitude_max_v=float(amp.max()) if len(amp) else None,
                tile_counts=counts, noise=quality)
            report['conditions'].setdefault(scenario,{})[role] = row
            values[scenario,role] = dict(phase=ph, amp=amp, mapped=mapped,
                                        waveform=envelope(signal, fs, first, last))
            print(scenario, role, row, flush=True)
            del signal
        del clean
    args.output.mkdir(parents=True,exist_ok=False)
    fig, axes = plt.subplots(4,6,figsize=(24,13),squeeze=False)
    for col,role in enumerate(['uhf','hfct']):
        raw_top = max((float(v['amp'].max()) for (s,r),v in values.items() if r==role and len(v['amp'])),default=refs[role])
        wave_top = max((float(np.max(np.abs(np.r_[v['waveform'][1],v['waveform'][2]]))) for (s,r),v in values.items() if r==role),default=1.)
        for row,scenario in enumerate(scenarios):
            v = values[scenario,role]
            ax = axes[row,col]
            t,low,high = v['waveform']
            ax.fill_between(t,low,high,color='steelblue',alpha=.7)
            ax.set_ylim(-1.05*wave_top,1.05*wave_top)
            ax.set_title(f'{role} waveform: {scenario}')
            ax.set_xlabel('Time within cycle (ms)');ax.set_ylabel('V')
            sample=np.linspace(0,len(v['amp'])-1,min(20000,len(v['amp'])),dtype=int) if len(v['amp']) else np.array([],dtype=int)
            for offset,key,top in [(2,'amp',raw_top*1.05),(4,'mapped',1.)]:
                ax=axes[row,col+offset]
                ax.scatter(v['phase'][sample],v[key][sample],s=.5,alpha=.6)
                ax.set_xlim(0,360);ax.set_ylim(0,top if top else 1)
                if key=='amp':ax.axhline(refs[role],color='red',ls='--',lw=.7)
                ax.set_title(f'{role}: '+('unclipped amplitude (V)' if key=='amp' else 'normalized/clipped amplitude'))
                ax.set_xlabel('Phase (degrees)')
    fig.suptitle(parent_id+' | diagnostic plots, not model inputs')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(args.output/'waveform_prpd.png',dpi=140);plt.close(fig)
    canvas=Image.new('RGB',(1344,8*150),'white');draw=ImageDraw.Draw(canvas)
    for row,scenario in enumerate(scenarios):
        for k,role in enumerate(['uhf','hfct']):
            y=(row*2+k)*150
            draw.text((5,y+3),f'{scenario} | {role} | pulses={len(values[scenario,role]["amp"])}',fill='black')
            for i,tile in enumerate(tiles):
                path=image_paths(args.data,scenario,tile)[k]
                with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((112,112)),(i*112,y+26))
                draw.rectangle((i*112,y+26,(i+1)*112-1,y+137),outline='gray')
    canvas.save(args.output/'existing_tiles.png')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved two PNGs and report.json:',args.output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--class-name',default='jd')
    parser.add_argument('--record-id')
    main(parser.parse_args())
