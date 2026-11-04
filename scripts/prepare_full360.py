"""Audit all 786 CH2 references, then create one complete 224px PRPD per sensor.

Inherit deduplicated split; fit per-channel axes on TRAIN peaks only. No
filtering, threshold changes, block splitting, training or noisy image creation.
"""
import argparse
import copy
import csv
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import generate_prpd_wfm as g
from revision_data import read_parent_record, legacy_pulses, image_paths


def original_records(m):
    if m.get('representation') not in (None,'original_records_for_full_prpd','whole_record_prpd'):
        raise ValueError('Provide the deduplicated original-record manifest, not tiles/regions')
    records=m['records']
    if len(records)!=786 or any('parent_record_id' in r for r in records):raise ValueError('Expected 786 original captures')
    if len({r['record_id'] for r in records})!=786 or len({r['group_id'] for r in records})!=786:raise ValueError('Duplicate IDs/groups')
    if dict(Counter(r['split'] for r in records))!={'train':471,'val':157,'test':158}:raise ValueError('Unexpected split counts')
    excluded=set(m.get('deduplication',{}).get('excluded_record_ids',[]))
    if excluded & {r['record_id'] for r in records}:raise ValueError('Excluded duplicate remains')
    return records


def main(args):
    path=args.manifest;raw=path.read_bytes();m=json.loads(raw)
    records=original_records(m)
    if args.output.exists():raise FileExistsError('Use a fresh output directory')
    if 'armed' not in inspect.getsource(g.legacy_compute_phase_from_ch2):raise ValueError('Corrected CH2 entry not installed')
    tracked=[ROOT/'generate_prpd_wfm.py',ROOT/'revision_data.py',ROOT/'mso44_wfm.py',ROOT/'mso44_csv.py']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    report=dict(source_manifest_sha256=hashlib.sha256(raw).hexdigest(),code_sha256=hashes,
        criteria='Corrected CH2, finite samples, each measured period 15-25 ms; no silent exclusion.',
        amplitude_normalization=False,records=[],all_passed=False)
    args.output.mkdir(parents=True,exist_ok=False)
    failures=[]
    for i,r in enumerate(records,1):
        entry={k:r[k] for k in ['record_id','class_name','split']}
        waves=phase=crossings=None
        try:
            waves=read_parent_record(args.raw_data,r)
            wave=waves['phase_ref']
            if not np.isclose(wave.info.sample_rate_hz,r['sample_rate_hz'],rtol=1e-10,atol=0):raise ValueError('CH2 fs mismatch')
            if 'samples' in r and wave.info.samples!=r['samples']:raise ValueError('CH2 length mismatch')
            phase,crossings=g.legacy_compute_phase_from_ch2(wave.values(),r['sample_rate_hz'],smooth_win=21)
            periods=np.diff(crossings)*1000
            if len(periods)<1 or not np.all(np.isfinite(periods)) or np.any((periods<15)|(periods>25)):raise ValueError('Invalid periods')
            entry.update(status='PASS',complete_cycles=len(periods),min_ms=float(periods.min()),
                         median_ms=float(np.median(periods)),max_ms=float(periods.max()))
            print(f'CH2 [{i}/786] PASS {r["record_id"]}: {entry["min_ms"]:.6f}/{entry["median_ms"]:.6f}/{entry["max_ms"]:.6f} ms',flush=True)
        except Exception as exc:
            entry.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));failures.append(entry)
            print(f'CH2 [{i}/786] FAIL {r["record_id"]}: {exc}',flush=True)
        finally:del waves,phase,crossings
        report['records'].append(entry)
        report.update(checked=i,failed=len(failures),passed=i-len(failures))
        if i%10==0 or i==786:
            (args.output/'phase_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['all_passed']=not failures
    audit=args.output/'phase_audit.json'
    audit.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output/'abnormal_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['record_id','class_name','split','status','error_type','error']);w.writeheader();w.writerows(failures)
    if failures:raise SystemExit(f'{len(failures)} abnormal records listed. No manifest/images created, no records excluded.')
    if path.read_bytes()!=raw or any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[p.name] for p in tracked):
        raise ValueError('Code/manifest changed during audit; rendering stopped')
    cache={};maxima={'uhf':0.,'hfct':0.}
    train=[r for r in records if r['split']=='train']
    for i,r in enumerate(train,1):
        pulses=legacy_pulses(args.raw_data,r);cache[r['record_id']]=pulses
        for role in maxima:
            amp=pulses[role][1]
            if not np.all(np.isfinite(amp)):raise ValueError('Nonfinite detected amplitude')
            if len(amp):maxima[role]=max(maxima[role],float(amp.max()))
        print(f'Fit TRAIN axis [{i}/471] {r["record_id"]}',flush=True)
    if any(v<=0 for v in maxima.values()):raise ValueError('No positive training peaks')
    axes={role:1.05*v for role,v in maxima.items()}
    new=copy.deepcopy(m);new.update(schema=4,representation='whole_record_prpd',records=records,parent_records=786)
    new['preprocessing']=copy.deepcopy(m.get('preprocessing',{}))
    for key in ['phase_tiles','tile_degrees','a_ref','reference_fit_split','reference_quantile','reference_scope']:new['preprocessing'].pop(key,None)
    new['preprocessing'].update(phase_method='hysteresis_rising_zero_v1',phase_period_guard_ms=[15,25],
        global_view_degrees=360,model_input_resize_px=[224,224],
        rendering='Complete360, raw volts, fixed per-channel TRAIN axes, scatter size8 alpha.9, resize224',
        amplitude='Raw absolute detected voltage in V',amplitude_normalization=False,amplitude_clipping=False,
        phase_smoothing=21,highpass=False,envelope=False,threshold='median(abs(x))+8*MAD(abs(x))',
        peak_group_gap_us=2,y_axis_fit_split='train',y_axis_policy='1.05 * max detected clean train peak per channel',
        train_peak_max_v=maxima,y_axis_max_v=axes)
    new['phase_revision']=dict(source_original_manifest_sha256=hashlib.sha256(raw).hexdigest(),code_sha256=hashes,
        audit_report_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
    new['amplitude_revision']=dict(phase_code_sha256=hashes['generate_prpd_wfm.py'],train_records=471,
        source_manifest_sha256=hashlib.sha256(raw).hexdigest(),train_peak_max_v=maxima,y_axis_max_v=axes,
        note='No amplitude mapping. Out-of-axis peaks are outside image and explicitly counted.')
    (args.output/'manifest.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
    print('All 786 CH2 references passed. Rendering COMPLETE 360deg images, NOT tiles.',flush=True)
    for i,r in enumerate(records,1):
        pulses=cache.pop(r['record_id']) if r['record_id'] in cache else legacy_pulses(args.raw_data,r)
        quality=dict(record_id=r['record_id'],split=r['split'],channels={})
        for role,target in zip(['uhf','hfct'],image_paths(args.output,'clean',r)):
            ph,amp=pulses[role]
            g.legacy_save_scatter_tile(target,ph,amp,xlim=(0.,360.),ylim=(0.,axes[role]),
                s=8,alpha=.9,dpi=256,bg_white=True,resize_to=224)
            quality['channels'][role]=dict(detections=len(amp),empty=len(amp)==0,outside_axis=int(np.sum(amp>axes[role])),y_axis_max_v=axes[role])
        q=args.output/'images'/'clean'/'quality'/(r['record_id']+'.json');q.parent.mkdir(parents=True,exist_ok=True)
        q.write_text(json.dumps(quality,indent=2),encoding='utf-8')
        print(f'Full360 [{i}/786] {r["record_id"]}: one UHF + one HFCT image',flush=True)
    if len(list((args.output/'images'/'clean').rglob('*.png')))!=1572:raise ValueError('Incomplete image count')
    print('DONE: 786 paired records, 1572 full360 images; train471/val157/test158. No training/noise.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--raw-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
