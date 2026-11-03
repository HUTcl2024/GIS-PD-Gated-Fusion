"""Auditable WFM/CSV -> paired PRPD -> fixed independent-record split.

CH1/UHF, CH2/phase, CH3/HFCT confirmed by author on 2026-10-27.
Author states every record is an independent experiment. This provenance is
an author declaration, not something filenames or waveform headers can prove.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from mso44_wfm import read_triplet

ROOT = Path(__file__).resolve().parent
CLASSES = ['jd', 'qx', 'xf', 'ym']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def assign_record_splits(records,seed):
    """Split original captures, not views; preserve input ordering and identities."""
    from sklearn.model_selection import train_test_split
    if not 0<=seed<2**32: raise ValueError('split-seed must be in [0, 2**32)')
    if (len({r['group_id'] for r in records})!=len(records)
            or len({r['record_id'] for r in records})!=len(records)):
        raise ValueError('Original records and group IDs must be unique')
    labels=[r['class_name'] for r in records]
    train,hold=train_test_split(np.arange(len(records)),test_size=.4,stratify=labels,random_state=seed)
    val,test=train_test_split(hold,test_size=.5,stratify=[labels[i] for i in hold],random_state=seed)
    result=[dict(r) for r in records]
    for name,indices in [('train',train),('val',val),('test',test)]:
        for i in indices: result[int(i)]['split']=name
    return result


def audit(args):
    from sklearn.model_selection import train_test_split
    out = args.output
    if (out / 'manifest.json').exists():
        raise FileExistsError('Manifest exists; use a new output directory.')
    records, errors = [], []
    for cls in CLASSES:
        folder = args.data / cls
        channels = {c: {p.name[:-8]: p for p in folder.glob(f'*_ch{c}.wfm')} for c in (1, 2, 3)}
        for stem in sorted(set().union(*(set(v) for v in channels.values()))):
            missing = [c for c in channels if stem not in channels[c]]
            if missing:
                errors.append({'record_id': stem, 'error': f'Missing channels {missing}'})
                continue
            try:
                paths = [channels[c][stem] for c in (1, 2, 3)]
                waves = read_triplet(*paths)
                if args.checksum and not all(w.verify_checksum() for w in waves.values()):
                    raise ValueError('Checksum failed')
                info = waves['uhf'].info
                if info.sample_rate_hz <= 2e6:
                    raise ValueError('Sample rate incompatible with 1 MHz high-pass')
                if any(w.info.y_unit != 'V' for w in waves.values()):
                    raise ValueError('Expected voltage units for all channels')
                v = re.search(r'-(\d+(?:\.\d+)?)kv', stem, re.I)
                records.append(dict(record_id=stem, class_name=cls, source_format='wfm',
                    group_id=f'{cls}/{stem}', voltage_kv=float(v[1]) if v else None,
                    sources=[p.relative_to(args.data).as_posix() for p in paths], samples=info.samples,
                    sample_rate_hz=info.sample_rate_hz, duration_s=info.samples*info.dt_s,
                    checksum_verified=args.checksum))
            except Exception as e:
                errors.append({'record_id': stem, 'error': str(e)})
        csv_hashes=set()
        from mso44_csv import CsvRecord
        for path in sorted(folder.glob('*.csv')):
            try:
                record=CsvRecord(path)
                if not np.isclose(record.info.samples*record.info.dt_s,1.,rtol=0,atol=1e-6):
                    raise ValueError('CSV must have a 1-second observation window, matching the WFM protocol')
                if record.info.sample_rate_hz<=2e6:
                    raise ValueError('CSV sample rate incompatible with 1 MHz high-pass')
                crossings=record.check_phase()
                if record.sha256 in csv_hashes:
                    raise ValueError('Duplicate CSV file content')
                csv_hashes.add(record.sha256)
                stem=path.stem
                if any(r['class_name']==cls and r['record_id']==stem for r in records):
                    raise ValueError('CSV and WFM share a record ID; check for duplicate exports')
                v=re.search(r'(\d+(?:\.\d+)?)\s*kv',stem,re.I)
                records.append(dict(record_id=stem,class_name=cls,group_id=f'{cls}/{stem}',
                    source_format='csv',voltage_kv=float(v[1]) if v else None,
                    sources=[path.relative_to(args.data).as_posix()],samples=record.info.samples,
                    sample_rate_hz=record.info.sample_rate_hz,duration_s=1.,
                    checksum_verified=None,source_sha256=record.sha256,phase_crossings=crossings))
                del record
            except Exception as e:
                errors.append({'record_id':path.stem,'error':str(e)})
    dump(out/'audit_errors.json', errors)
    if errors:
        raise ValueError(f'{len(errors)} invalid records; fix or explicitly exclude them before freezing the manifest. See audit_errors.json')
    if len({r['group_id'] for r in records}) != len(records):
        raise ValueError('Duplicate independent group IDs')
    records=assign_record_splits(records,args.split_seed)
    manifest = dict(schema=2, classes=CLASSES, split_seed=args.split_seed,
        provenance='Author confirms each WFM triplet and each added three-channel CSV is an independent acquisition, not another export of a WFM, 2026-10-27; author declaration, not independently verified.',
        records=records)
    dump(out/'manifest.json', manifest)
    with (out/'acquisition_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['record_id','class_name','group_id','voltage_kv','split','specimen_id','session_id'])
        writer.writeheader()
        for r in records:
            writer.writerow({k:r.get(k,'') for k in writer.fieldnames})
    print(json.dumps({s:dict(Counter(r['class_name'] for r in records if r['split']==s))
                      for s in ['train','val','test']},ensure_ascii=False))


def corrupt(signal, fraction, rng):
    """Composite: white RMS=a*clean RMS; pulse peak=a*clean abs peak.

    Ten signed Gaussian pulses per second, sigma=1 us. Not a reconstruction
    of the unspecified historical noise experiment; explicit new protocol.
    """
    signal = signal - np.median(signal)
    if fraction == 0:
        return signal
    noise = rng.standard_normal(len(signal))
    noise *= fraction * np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(noise**2))
    return signal + noise  # Gaussian pulses added by render using actual fs.


def image_paths(out, scenario, record):
    return [out/'images'/scenario/role/record['class_name']/(record['record_id']+'.png')
            for role in ['uhf','hfct']]


REGION_VIEWS=[('global',0,360),('p000_120',0,120),('p120_240',120,240),('p240_360',240,360)]


def region_paths(out,scenario,record):
    return [out/'images'/scenario/role/record['class_name']/(record['record_id']+'_'+view+'.png')
        for role in ['uhf','hfct'] for view,_,_ in REGION_VIEWS]


def region_resplit(args):
    """New record split and new train-only references; never reuse old images."""
    if args.parent_manifest is None: raise ValueError('Provide existing region manifest')
    if args.output.exists(): raise FileExistsError('Choose a new, nonexistent output directory')
    if args.smoke or args.limit or args.record_ids or args.fraction or args.snr_db is not None or args.train_bank:
        raise ValueError('Resplitting/refitting requires all parent records and clean waveforms')
    raw=args.parent_manifest.read_bytes(); old=json.loads(raw)
    if old.get('representation')!='legacy_global_three_regions': raise ValueError('Expected original-record region manifest')
    if any('parent_record_id' in r for r in old['records']): raise ValueError('Do not split child tiles')
    records=assign_record_splits(old['records'],args.split_seed)
    if set(r['class_name'] for r in records)!=set(old['classes']): raise ValueError('Class set mismatch')
    counts={s:dict(Counter(r['class_name'] for r in records if r['split']==s)) for s in ['train','val','test']}
    changed=sum(a['split']!=b['split'] for a,b in zip(old['records'],records))
    print(f'NEW split seed={args.split_seed}; parents={len(records)}; changed memberships={changed}. Old files untouched.',flush=True)
    print(json.dumps(counts,ensure_ascii=False),flush=True)
    train=[r for r in records if r['split']=='train']; amplitudes={role:[] for role in ['uhf','hfct']}
    for i,r in enumerate(train,1):
        pulses=legacy_pulses(args.data,r)
        for role in amplitudes:
            if pulses[role][1].size: amplitudes[role].append(pulses[role][1])
        print(f'fit NEW train only [{i}/{len(train)}] {r["record_id"]}',flush=True)
    references={}
    for role,parts in amplitudes.items():
        if not parts: raise ValueError('No training pulses for '+role)
        references[role]=float(np.quantile(np.concatenate(parts),.99))
        if not np.isfinite(references[role]) or references[role]<=0: raise ValueError('Invalid reference')
    manifest=dict(old,split_seed=args.split_seed,records=records,
        source_region_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_split_seed=old.get('split_seed'),
        split_protocol='two-stage stratified 60/20/20 by original record; both stages use split_seed; inherited record order',
        evaluation_warning='New development split of existing records, not newly acquired independent or untouched test data')
    for key in ['source_tile_manifest_sha256','parent_manifest_sha256']: manifest.pop(key,None)
    manifest['preprocessing']=dict(old['preprocessing'],a_ref=references,reference_fit_split='train',reference_quantile=.99)
    dump(args.output/'manifest.json',manifest)
    print(f'New region manifest saved; train-only references={references}. Render new images and retrain all compared models.',flush=True)


def region_fit(args):
    if args.parent_manifest is None: raise ValueError('Provide the existing tile manifest with fitted references')
    if (args.output/'manifest.json').exists(): raise FileExistsError('Region manifest already exists')
    raw=args.parent_manifest.read_bytes(); old=json.loads(raw)
    if old.get('representation')!='legacy_phase_tiles_30deg': raise ValueError('Expected existing 12-tile manifest')
    grouped={}
    for r in old['records']: grouped.setdefault(r['parent_record_id'],[]).append(r)
    records=[]
    for parent,children in grouped.items():
        if (len(children)!=12 or len({r['split'] for r in children})!=1 or len({r['class_name'] for r in children})!=1
                or {r['phase_start_deg'] for r in children}!=set(range(0,360,30))):
            raise ValueError('Invalid inherited parent group')
        record={k:v for k,v in children[0].items() if k not in ['parent_record_id','phase_start_deg','phase_stop_deg']}
        record['record_id']=parent; records.append(record)
    if len({r['group_id'] for r in records})!=len(records): raise ValueError('Duplicate parent group')
    manifest=dict(old,representation='legacy_global_three_regions',records=records,
        source_tile_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        views=[dict(name=name,start_deg=a,stop_deg=b) for name,a,b in REGION_VIEWS],
        input_unit='one original record; 4 views per channel',
        design='three equal 120-degree regions, an adaptation rather than verified reproduction of a published architecture')
    manifest['preprocessing']={k:v for k,v in old['preprocessing'].items() if k not in ['phase_tiles','tile_degrees']}
    manifest['preprocessing'].update(global_view_degrees=360,local_regions=3,local_region_degrees=120)
    dump(args.output/'manifest.json',manifest)
    print(f'Locked {len(records)} original records; inherited splits and unchanged amplitude references.',flush=True)


def region_render(args):
    from generate_prpd_wfm import legacy_normalize_amplitude_global,legacy_save_scatter_tile
    m=json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    if m.get('representation')!='legacy_global_three_regions': raise ValueError('Expected region manifest')
    if args.fraction or args.train_bank: raise ValueError('Region renderer supports clean or specified AWGN SNR only')
    noisy=args.snr_db is not None
    scenario=f'awgn_{args.target}_snr{args.snr_db:g}dB' if noisy else 'clean'
    selected=[args.noise_split] if noisy else args.render_splits
    if noisy and args.noise_split=='val': scenario='val_'+scenario
    config=dict(split=selected[0] if noisy else selected,snr_db=args.snr_db,target=args.target if noisy else None,
        noise='AWGN' if noisy else 'none',noise_seed=2026,phase_reference='unperturbed CH2',
        amplitude_reference=m['preprocessing']['a_ref'],views=m['views'],
        manifest_sha256=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest())
    path=args.output/'images'/scenario/'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config: raise ValueError('Region scenario configuration mismatch')
    records=[r for r in m['records'] if r['split'] in selected]
    if args.record_ids:
        if not set(args.record_ids)<={r['record_id'] for r in records}: raise ValueError('Unknown selected record')
        records=[r for r in records if r['record_id'] in args.record_ids]
    if args.smoke:
        records=[next(r for r in records if r['split']==s and r['class_name']==c) for s in selected for c in m['classes']]
    if args.limit: records=records[:args.limit]
    dump(path,config)
    for i,r in enumerate(records,1):
        quality=args.output/'images'/scenario/'quality'/(r['record_id']+'.json')
        if quality.exists() and all(p.is_file() for p in region_paths(args.output,scenario,r)): continue
        noise_quality={}; pulses=legacy_pulses(args.data,r,target=args.target,snr_db=args.snr_db,noise_quality=noise_quality)
        paths=region_paths(args.output,scenario,r); counts={}
        for k,role in enumerate(['uhf','hfct']):
            phase,amp=pulses[role]
            norm=legacy_normalize_amplitude_global(amp,m['preprocessing']['a_ref'][role],mode='asinh',gamma=2.)
            for j,(view,a,b) in enumerate(REGION_VIEWS):
                mask=(phase>=a)&((phase<b) if b<360 else (phase<=360))
                legacy_save_scatter_tile(paths[k*4+j],phase[mask]-a,norm[mask],xlim=(0.,float(b-a)))
                counts.setdefault(role,{})[view]=int(mask.sum())
        dump(quality,dict(record_id=r['record_id'],split=r['split'],counts=counts,noise=noise_quality))
        print(f'region views [{i}/{len(records)}] {scenario} {r["record_id"]}: 8 images',flush=True)


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
        elif fraction and target in [role,'both']:
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'2026/{record["record_id"]}/{role}'.encode()).digest()[:8],'little'))
            clean=signal-np.median(signal)
            signal=corrupt(signal,fraction,rng)
            width=max(1,int(round(fs*1e-6))); offsets=np.arange(-5*width,5*width+1)
            pulse=np.exp(-.5*(offsets/width)**2)*fraction*np.max(np.abs(clean))
            for center in rng.integers(0,len(signal),size=max(1,round(len(signal)/fs*10))):
                indices=center+offsets; mask=(indices>=0)&(indices<len(signal))
                signal[indices[mask]]+=rng.choice([-1,1])*pulse[mask]
        result[role]=legacy_detect_pulse_peaks(signal,phase,fs,win_us=2.,thr_k=8.)[:2]
    return result


def online_parent_images(root,record,references,snr_db,target,noise_seed,view_definitions=None,pulses_override=None):
    """Legacy rendering to compressed PNG bytes in RAM, with no file writes."""
    from io import BytesIO
    from PIL import Image
    from generate_prpd_wfm import legacy_normalize_amplitude_global,plt
    pulses=(legacy_pulses(root,record,target=target,snr_db=snr_db,noise_seed=noise_seed)
        if pulses_override is None else pulses_override)
    result={}
    for role in pulses:
        phase,amp=pulses[role]
        amplitude=legacy_normalize_amplitude_global(amp,references[role],mode='asinh',gamma=2.)
        views=view_definitions if view_definitions is not None else [(f'p{a:03d}_{a+30:03d}',a,a+30) for a in range(0,360,30)]
        for view,start,stop in views:
            mask=(phase>=start)&((phase<stop) if stop<360 else (phase<=stop))
            fig=plt.figure(figsize=(2.56,2.56),dpi=256); ax=fig.add_axes([0,0,1,1])
            try:
                fig.patch.set_facecolor('white'); ax.set_facecolor('white')
                if mask.any(): ax.scatter(phase[mask]-start,amplitude[mask],s=8,marker='.',linewidths=0,alpha=.9)
                ax.set_xlim(0.,float(stop-start)); ax.set_ylim(0.,1.); ax.set_axis_off()
                buffer=BytesIO(); fig.savefig(buffer,format='png',dpi=256,bbox_inches='tight',pad_inches=0)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    resized=image.resize((224,224),resample=Image.Resampling.LANCZOS)
                    encoded=BytesIO(); resized.save(encoded,format='PNG')
                rid=f'{record["record_id"]}_{view}'
                result.setdefault(rid,{})[role]=encoded.getvalue()
            finally: plt.close(fig)
    return result


def tile_fit(args):
    """Fit legacy amplitude references on train parents only; lock inherited split."""
    if args.parent_manifest is None: raise ValueError('--parent-manifest is required')
    if args.smoke or args.limit or args.record_ids or args.fraction:
        raise ValueError('Normalization fitting must use every training parent, clean only')
    if (args.output/'manifest.json').exists(): raise FileExistsError('Tile manifest already exists; do not refit')
    raw=args.parent_manifest.read_bytes(); parent=json.loads(raw)
    if parent.get('representation'): raise ValueError('Expected original-record parent manifest')
    records=parent['records']
    if len({r['group_id'] for r in records})!=len(records): raise ValueError('Parent group IDs must be unique')
    train=[r for r in records if r['split']=='train']
    amplitudes={role:[] for role in ['uhf','hfct']}
    for i,r in enumerate(train,1):
        pulses=legacy_pulses(args.data,r)
        for role in amplitudes:
            if pulses[role][1].size: amplitudes[role].append(pulses[role][1])
        print(f'fit legacy train only [{i}/{len(train)}] {r["record_id"]}',flush=True)
    references={}
    for role,parts in amplitudes.items():
        if not parts: raise ValueError('No training pulses for '+role)
        references[role]=float(np.quantile(np.concatenate(parts),.99))
        if not references[role]>0: raise ValueError('Invalid training amplitude reference')
    tiles=[]
    for r in records:
        for a in range(0,360,30):
            t=dict(r); t.update(record_id=f'{r["record_id"]}_p{a:03d}_{a+30:03d}',
                parent_record_id=r['record_id'],phase_start_deg=a,phase_stop_deg=a+30)
            tiles.append(t)
    manifest=dict(schema=3,classes=parent['classes'],split_seed=parent['split_seed'],
        representation='legacy_phase_tiles_30deg',parent_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        parent_records=len(records),provenance=parent['provenance'],
        preprocessing=dict(phase_smoothing=21,threshold='median(abs(signal))+8*MAD(abs(signal))',
            peak_group_gap_us=2,amplitude='asinh(2*amp/a_ref)/asinh(2), clipped to [0,1]',
            a_ref=references,reference_fit_split='train',reference_quantile=.99,
            reference_scope='per-channel; consistent with legacy dual-channel prpd.py',
            rendering='white background; default blue scatter, size 8, alpha .9; 2.56 inch, dpi 256; LANCZOS resize 224',
            phase_tiles=12,tile_degrees=30,highpass=False,envelope=False),records=tiles)
    dump(args.output/'manifest.json',manifest)
    print(f'Locked {len(records)} parent records -> {len(tiles)} paired tiles; references={references}',flush=True)


def tile_render(args):
    from generate_prpd_wfm import legacy_normalize_amplitude_global,legacy_save_scatter_tile
    m=json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    if m.get('representation')!='legacy_phase_tiles_30deg': raise ValueError('Expected legacy tile manifest')
    grouped={}
    for r in m['records']: grouped.setdefault(r['parent_record_id'],[]).append(r)
    parents=list(grouped)
    bank=getattr(args,'train_bank',False)
    if bank and (args.fraction or args.snr_db is not None or args.smoke or args.limit or args.record_ids):
        raise ValueError('train-bank requires the complete clean training split, without other noise or subset options')
    snr_db=getattr(args,'snr_db',None)
    scenario=(f'awgn_{args.target}_snr{snr_db:g}dB' if snr_db is not None else
              ('clean' if not args.fraction else f'composite_{args.target}_{args.fraction:g}'))
    if bank:
        scenario='train_awgn_bank_v1'
        parents=[p for p in parents if grouped[p][0]['split']=='train']
        plan={}
        for parent in parents:
            seed=int.from_bytes(hashlib.sha256(f'3407/train-bank/{parent}'.encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            plan[parent]=dict(snr_db=float(rng.uniform(-15,10)),target=str(rng.choice(['both','uhf','hfct'])))
        config=dict(noise='AWGN',split='train',planning_seed=3407,snr_distribution='uniform continuous [-15,10] dB',
            target_distribution='uniform both/uhf/hfct per parent',plan=plan,
            noise_seed=2026,parent_manifest_sha256=m['parent_manifest_sha256'],
            tile_manifest_sha256=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest(),
            preprocessing='same legacy pulse detection and fixed training-only amplitude references; unperturbed CH2',
            snr_definition='10*log10(mean((clean-median(clean))**2)/mean(added_noise**2))',
            warning='One fixed noise realization, SNR and target per training parent; not resampled each epoch')
        path=args.output/'images'/scenario/'config.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config: raise ValueError('Training bank configuration mismatch')
        dump(path,config)
    elif args.fraction or snr_db is not None:
        noise_split=getattr(args,'noise_split','test')
        parents=[p for p in parents if grouped[p][0]['split']==noise_split]
        if noise_split=='val': scenario='val_'+scenario
        config=dict(fraction=args.fraction,target=args.target,split=noise_split,domain='raw waveform before legacy pulse extraction',
            white_rms='fraction * RMS(median-centered clean channel)',pulse_peak='fraction * peak(abs(median-centered clean channel))',
            pulse_sigma_s=1e-6,pulses_per_second=10,noise_seed=2026,phase_reference='unperturbed CH2',
            amplitude_reference='unchanged training-only per-channel a_ref',parent_manifest_sha256=m['parent_manifest_sha256'])
        if snr_db is not None:
            config=dict(noise='AWGN',snr_db=snr_db,target=args.target,split=noise_split,
                snr_definition='10*log10(mean((clean-median(clean))**2)/mean(added_noise**2))',
                power_scope='entire captured waveform including existing measurement noise; not isolated PD power',
                scaling='zero-mean Gaussian realization rescaled per original record and channel to requested measured SNR',
                domain='raw waveform before legacy pulse extraction',noise_seed=2026,
                pairing='same standardized noise per parent/channel across SNRs and methods',
                phase_reference='unperturbed CH2',amplitude_reference='unchanged training-only per-channel a_ref',
                parent_manifest_sha256=m['parent_manifest_sha256'])
        path=args.output/'images'/scenario/'config.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config: raise ValueError('Noise configuration mismatch')
        dump(path,config)
    if args.record_ids:
        if not set(args.record_ids)<=set(parents): raise ValueError('Unknown parent record IDs')
        parents=[p for p in parents if p in args.record_ids]
    if args.smoke:
        parents=[p for s in ['train','val','test'] for c in m['classes']
                 for p in [p for p in parents if grouped[p][0]['split']==s and grouped[p][0]['class_name']==c][:1]]
    if args.limit: parents=parents[:args.limit]
    for i,parent_id in enumerate(parents,1):
        tiles=grouped[parent_id]
        if len(tiles)!=12 or len({t['split'] for t in tiles})!=1: raise ValueError('Invalid parent tile group')
        quality=args.output/'images'/scenario/'quality'/(parent_id+'.json')
        if quality.exists() and all(p.exists() for t in tiles for p in image_paths(args.output,scenario,t)): continue
        source=dict(tiles[0],record_id=parent_id)
        noise_quality={}
        local_snr=plan[parent_id]['snr_db'] if bank else snr_db
        local_target=plan[parent_id]['target'] if bank else args.target
        pulses=legacy_pulses(args.data,source,args.fraction,local_target,local_snr,noise_quality); counts={}
        for role in ['uhf','hfct']:
            phase,amp=pulses[role]
            normalized=legacy_normalize_amplitude_global(amp,m['preprocessing']['a_ref'][role],mode='asinh',gamma=2.)
            for t in tiles:
                a,b=t['phase_start_deg'],t['phase_stop_deg']
                mask=(phase>=a)&((phase<b) if b<360 else (phase<=b))
                index=0 if role=='uhf' else 1
                path=image_paths(args.output,scenario,t)[index]
                legacy_save_scatter_tile(path,phase[mask]-a,normalized[mask],xlim=(0.,30.),
                    ylim=(0.,1.),s=8,alpha=.9,dpi=256,bg_white=True,resize_to=224)
                counts.setdefault(t['record_id'],{})[role]=int(mask.sum())
        dump(quality,dict(parent_record_id=parent_id,split=source['split'],tile_peak_counts=counts,noise= noise_quality))
        print(f'legacy tiles [{i}/{len(parents)}] {scenario} {parent_id}: 12 paired blocks',flush=True)


def render(args):
    from generate_prpd_wfm import get_phase_from_ch2, extract_pulses, draw_prpd_224
    manifest=json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    scenario = 'clean' if args.fraction == 0 else f'composite_{args.target}_{args.fraction:g}'
    config=dict(fraction=args.fraction,target=args.target,pulse_sigma_s=1e-6,
                pulses_per_second=10,noise_seed=2026,
                white_scale='RMS of median-centered clean channel',
                pulse_scale='peak absolute median-centered clean channel',
                display_scale='clean 99th percentile detected pulse amplitude, fixed across scenarios',
                phase='CH2 positive zero crossings; samples outside first/last crossing excluded',
                highpass_hz=[1000000,500000],threshold_mad=6,peak_distance_s=1e-5)
    config_path=args.output/'images'/scenario/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing scenario has different settings')
    dump(config_path,config)
    records=manifest['records']
    if args.fraction:
        records=[r for r in records if r['split']=='test']
    if args.limit:
        records=records[:args.limit]
    if args.smoke:
        records=[r for split in ['train','val','test'] for cls in CLASSES
                 for r in [x for x in records if x['split']==split and x['class_name']==cls][:2]]
    if args.record_ids:
        requested=set(args.record_ids)
        records=[r for r in records if r['record_id'] in requested]
        if {r['record_id'] for r in records}!=requested:
            raise ValueError('Some requested records are absent from this selection')
    for number,r in enumerate(records,1):
        targets=image_paths(args.output,scenario,r)
        quality=args.output/'images'/scenario/'quality'/(r['record_id']+'.json')
        if all(p.exists() for p in targets) and quality.exists():
            continue
        csv_record=None
        if r.get('source_format','wfm')=='csv':
            from mso44_csv import CsvRecord
            csv_record=CsvRecord(args.data/r['sources'][0])
            if csv_record.sha256!=r['source_sha256']:
                raise ValueError('CSV changed after manifest creation: '+r['record_id'])
            waves=csv_record.waves
            integrity=dict(source_format='csv',source_sha256_verified=True,instrument_checksum_verified=None)
        else:
            waves=read_triplet(*(args.data/p for p in r['sources']))
            if not all(w.verify_checksum() for w in waves.values()):
                raise ValueError('Checksum failed: '+r['record_id'])
            integrity=dict(source_format='wfm',source_sha256_verified=None,instrument_checksum_verified=True)
        if any(w.info.samples!=r['samples'] or not np.isclose(w.info.sample_rate_hz,r['sample_rate_hz']) for w in waves.values()):
            raise ValueError('Source grid differs from manifest')
        fs=r['sample_rate_hz']
        phase, smooth, freq=get_phase_from_ch2(waves['phase_ref'].times(),waves['phase_ref'].values(),fs)
        if phase is None:
            raise ValueError('Insufficient phase crossings: '+r['record_id'])
        zc=np.flatnonzero((smooth[:-1]<0)&(smooth[1:]>=0))+1
        valid=np.zeros(len(phase),dtype=bool); valid[zc[0]:zc[-1]]=True
        counts={}
        for role,hp,target in zip(['uhf','hfct'],[1000000,500000],targets):
            clean=waves[role].values()
            clean_peaks,clean_filtered=extract_pulses(clean,fs,hp,6.)
            clean_peaks=clean_peaks[valid[clean_peaks]]
            ymax=float(np.percentile(np.abs(clean_filtered[clean_peaks]),99))*1.1 if len(clean_peaks) else .1
            if not ymax>0: ymax=.1
            if args.fraction and args.target in [role,'both']:
                seed=int.from_bytes(hashlib.sha256(f'2026/{r["record_id"]}/{role}'.encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed)
                signal=corrupt(clean,args.fraction,rng)
                width=max(1,int(round(fs*1e-6))); offsets=np.arange(-5*width,5*width+1)
                pulse=np.exp(-.5*(offsets/width)**2)*args.fraction*np.max(np.abs(clean-np.median(clean)))
                for center in rng.integers(0,len(clean),size=max(1,round(len(clean)/fs*10))):
                    indices=center+offsets; mask=(indices>=0)&(indices<len(clean))
                    signal[indices[mask]]+=rng.choice([-1,1])*pulse[mask]
                peaks,filtered=extract_pulses(signal,fs,hp,6.)
                peaks=peaks[valid[peaks]]
            else:
                peaks,filtered=clean_peaks,clean_filtered
            target.parent.mkdir(parents=True,exist_ok=True)
            draw_prpd_224(phase[peaks],filtered[peaks],str(target),y_limit=ymax)
            counts[role]=len(peaks)
        dump(quality,dict(peak_counts=counts,reference_frequency_hz=freq,source_integrity=integrity))
        del waves,csv_record
        print(f'[{number}/{len(records)}] {scenario} {r["record_id"]} {counts}',flush=True)


def reliability_metrics(predictions):
    """Diagnostics from saved probabilities; oracle uses labels and is not deployable."""
    y=np.array([p['true'] for p in predictions]); pred=np.array([p['pred'] for p in predictions])
    probabilities=np.asarray([p['branch_probabilities'] for p in predictions],dtype=float)
    weights=np.asarray([p['weights'] for p in predictions],dtype=float)
    if (probabilities.ndim!=3 or probabilities.shape[:2]!=(len(y),2)
            or weights.shape!=(len(y),2) or not np.isfinite(probabilities).all() or not np.isfinite(weights).all()
            or (probabilities<0).any() or (weights<0).any()
            or not np.allclose(probabilities.sum(2),1,atol=1e-5) or not np.allclose(weights.sum(1),1,atol=1e-5)):
        raise ValueError('Invalid saved branch probabilities or weights')
    branch_pred=probabilities.argmax(2); confidence=probabilities.max(2)
    correct=branch_pred==y[:,None]; one=correct.sum(1)==1
    good_branch=correct[one].argmax(1)
    def select_rate(scores):
        selected=scores[one]; tie=np.abs(selected[:,0]-selected[:,1])<=1e-6
        hit=(selected.argmax(1)==good_branch)&~tie
        return (float(hit.mean()) if len(hit) else None,float(tie.mean()) if len(tie) else None)
    gate_hit,gate_tie=select_rate(weights); conf_hit,conf_tie=select_rate(confidence)
    wrong_conf=[float(confidence[~correct[:,k],k].mean()) if (~correct[:,k]).any() else None for k in [0,1]]
    return dict(accuracy=float((pred==y).mean()),uhf_accuracy=float(correct[:,0].mean()),hfct_accuracy=float(correct[:,1].mean()),
        oracle=float(correct.any(1).mean()),one_count=int(one.sum()),gate_hit=gate_hit,gate_tie=gate_tie,
        confidence_hit=conf_hit,confidence_tie=conf_tie,wrong_conf_uhf=wrong_conf[0],wrong_conf_hfct=wrong_conf[1])


def audit_reliability(args):
    if args.results is None: raise ValueError('audit-reliability requires --results')
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    raw=(args.output/'manifest.json').read_bytes(); m=json.loads(raw)
    protocol=read(args.results/'protocol.json')
    if protocol.get('evaluation_split')!='val' or protocol.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Expected matching validation result protocol')
    method='region_gfm' if 'region_gfm' in protocol['methods'] else 'gfm'
    seeds=protocol['seeds']; expected={}
    for r in m['records']:
        if r['split']=='val': expected[r.get('parent_record_id',r['record_id'])]=m['classes'].index(r['class_name'])
    scenarios=['clean']+[f'val_awgn_{target}_snr{snr}dB' for target in ['both','uhf','hfct'] for snr in [-15,-10,-5,0,5,10]]
    print(f'READ ONLY: {method}, seeds={seeds}, {len(expected)} validation records/seed.',flush=True)
    print('Oracle = either internal branch correct, uses true labels; diagnostic only, not a usable method or strict bound on fused logits.',flush=True)
    print('Gate/conf hit evaluated only when exactly one branch is correct; ties count separately.',flush=True)
    print('scenario | GFM acc% | UHF acc% | HFCT acc% | either correct% | mean one-correct N | gate hit% | gate tie% | confidence hit% | confidence tie% | wrong UHF confidence | wrong HFCT confidence',flush=True)
    for scenario in scenarios:
        metrics=[]
        for seed in seeds:
            d=read(args.results/f'{method}_seed{seed}'/f'val_{scenario}.json')
            rows=d['predictions']; actual={p['record_id']:p['true'] for p in rows}
            if len(rows)!=len(expected) or actual!=expected: raise ValueError(f'{seed} {scenario}: parent/label mismatch')
            metrics.append(reliability_metrics(rows))
        def mean(key,percent=True):
            values=[v[key] for v in metrics if v[key] is not None]
            return f'{np.mean(values)*(100 if percent else 1):.2f}' if values else 'NA'
        values=[mean(k) for k in ['accuracy','uhf_accuracy','hfct_accuracy','oracle']]
        values+=[mean('one_count',False)]
        values+=[mean(k) for k in ['gate_hit','gate_tie','confidence_hit','confidence_tie']]
        values+=[mean(k,False) for k in ['wrong_conf_uhf','wrong_conf_hfct']]
        print(scenario+' | '+' | '.join(values),flush=True)
    print('Seed-level means; missing conditional statistics omitted, NA means no eligible cases. Repeated seeds are not independent new records.',flush=True)
    print('No model/data/result changes; no files written.',flush=True)


def audit_gates(args):
    """Pair saved record predictions across methods, seeds and noise conditions."""
    if args.results is None: raise ValueError('audit-gates requires --results')
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    raw=(args.output/'manifest.json').read_bytes(); m=json.loads(raw)
    protocol=read(args.results/'protocol.json')
    if (protocol.get('evaluation_split')!='val' or
            protocol.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('Expected matching validation evaluation protocol')
    expected={}
    for r in m['records']:
        if r['split']=='val':
            rid=r['parent_record_id']; label=m['classes'].index(r['class_name'])
            if rid in expected and expected[rid]!=label: raise ValueError('Inconsistent parent label')
            expected[rid]=label
    if not expected: raise ValueError('No validation parents')
    classes=m['classes']; seeds=[42,43,44,45,46]
    scenarios=['clean']+[f'val_awgn_{target}_snr{db}dB'
        for target in ['both','uhf','hfct'] for db in [20,10,0,-5]]
    saved={}
    print(f'Reading saved predictions only: {len(expected)} validation parents, 5 seeds, equal versus GFM.',flush=True)
    for scenario in scenarios:
        for method in ['equal','gfm']:
            for seed in seeds:
                d=read(args.results/f'{method}_seed{seed}'/f'val_{scenario}.json')
                predictions=d['predictions']; rows={p['record_id']:p for p in predictions}
                if (d.get('metric_unit')!='original_record' or d['n']!=len(expected)
                        or len(rows)!=len(predictions) or set(rows)!=set(expected)):
                    raise ValueError(f'{method} {seed} {scenario}: parent set mismatch')
                for rid,p in rows.items():
                    if p['true']!=expected[rid] or not 0<=p['pred']<len(classes) or p.get('tile_count')!=12:
                        raise ValueError('Prediction label or tile count mismatch')
                    if method=='gfm':
                        w=np.asarray(p['weights'],dtype=float)
                        if w.shape!=(2,) or not np.isfinite(w).all() or (w<0).any() or (w>1).any() or abs(w.sum()-1)>1e-5:
                            raise ValueError('Invalid GFM weights')
                saved[method,seed,scenario]=rows
        print(f'Loaded {scenario}',flush=True)
    print('\nRECALL: record level; mean +/- sample SD across 5 seeds; percentage points for delta.',flush=True)
    print('scenario | class | records per seed | equal recall % | GFM recall % | paired GFM-equal pp | seeds GFM>equal',flush=True)
    for scenario in scenarios:
        for label,cls in enumerate(classes):
            ids=[rid for rid,y in expected.items() if y==label]
            if not ids: raise ValueError('Missing validation class')
            scores={method:np.array([100*np.mean([saved[method,s,scenario][rid]['pred']==label for rid in ids])
                for s in seeds]) for method in ['equal','gfm']}
            delta=scores['gfm']-scores['equal']
            fmt=lambda a:f'{np.mean(a):.2f}+/-{np.std(a,ddof=1):.2f}'
            print(f'{scenario} | {cls} | {len(ids)} | {fmt(scores["equal"])} | '
                f'{fmt(scores["gfm"])} | {fmt(delta)} | {int((delta>1e-9).sum())}/5',flush=True)
    print('\nGATES: weight of the CORRUPTED channel, averaged over 12 tiles per record.',flush=True)
    print('Negative delta means downweighting. Decrease/increase requires >=0.01 weight (1 percentage point).',flush=True)
    print('target | SNR | class | records per seed | clean weight | noisy weight | delta +/- SD | decrease % | stable % | increase %',flush=True)
    for role_index,role in enumerate(['uhf','hfct']):
        for db in [20,10,0,-5]:
            scenario=f'val_awgn_{role}_snr{db}dB'
            for cls in ['ALL']+classes:
                ids=[rid for rid,y in expected.items() if cls=='ALL' or classes[y]==cls]
                clean=np.array([[saved['gfm',s,'clean'][rid]['weights'][role_index] for rid in ids] for s in seeds])
                noisy=np.array([[saved['gfm',s,scenario][rid]['weights'][role_index] for rid in ids] for s in seeds])
                delta=noisy-clean; seed_delta=delta.mean(axis=1)
                down=100*float((delta<=-.01).mean()); up=100*float((delta>=.01).mean())
                stable=100*float(((delta>-.01)&(delta<.01)).mean())
                print(f'{role} | {db}dB | {cls} | {len(ids)} | {clean.mean():.4f} | '
                    f'{noisy.mean():.4f} | {seed_delta.mean():+.4f}+/-{seed_delta.std(ddof=1):.4f} | '
                    f'{down:.2f} | {stable:.2f} | {up:.2f}',flush=True)
    print('Gate fractions average paired record proportions across 5 seeds; these are not independent new records.',flush=True)
    print('Diagnostic associations only; lower weight alone does not prove a useful mechanism. No files written.',flush=True)


def audit_snr(args):
    """Read-only validation-wide pulse retention and amplitude clipping audit."""
    from generate_prpd_wfm import (legacy_compute_phase_from_ch2,
        legacy_detect_pulse_peaks,legacy_normalize_amplitude_global)
    if args.smoke or args.limit or args.record_ids or args.fraction:
        raise ValueError('audit-snr audits all validation parents, clean and 20/10/0/-5 dB')
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    m=read(args.output/'manifest.json')
    if m.get('representation')!='legacy_phase_tiles_30deg': raise ValueError('Expected legacy tile manifest')
    grouped={}
    for r in m['records']:
        if r['split']=='val': grouped.setdefault(r['parent_record_id'],[]).append(r)
    if not grouped: raise ValueError('No validation records')
    rows=[]; levels=[None,20,10,0,-5]
    print(f'Read-only audit: {len(grouped)} validation parents; no train/test waveforms or new files.',flush=True)
    for i,rid in enumerate(sorted(grouped),1):
        tiles=sorted(grouped[rid],key=lambda t:t['phase_start_deg'])
        if len(tiles)!=12 or {t['phase_start_deg'] for t in tiles}!=set(range(0,360,30)):
            raise ValueError(f'{rid}: invalid phase tiles')
        source=dict(tiles[0],record_id=rid)
        waves=read_parent_record(args.data,source); fs=source['sample_rate_hz']
        phase,_=legacy_compute_phase_from_ch2(waves['phase_ref'].values(),fs,smooth_win=21)
        for role in ['uhf','hfct']:
            clean=waves[role].values(); baseline=None
            for db in levels:
                signal=clean; measured=None
                if db is not None:
                    seed=int.from_bytes(hashlib.sha256(f'2026/awgn/{rid}/{role}'.encode()).digest()[:8],'little')
                    signal,_=add_awgn_snr(clean,db,np.random.default_rng(seed))
                    measured=float(10*np.log10(np.mean((clean-np.median(clean))**2)/np.mean((signal-clean)**2)))
                    if abs(measured-db)>1e-6: raise ValueError(f'{rid}: SNR mismatch')
                ph,amp=legacy_detect_pulse_peaks(signal,phase,fs,win_us=2.,thr_k=8.)[:2]
                counts=[int(((ph>=t['phase_start_deg'])&
                    ((ph<t['phase_stop_deg']) if t['phase_stop_deg']<360 else (ph<=360))).sum()) for t in tiles]
                count=sum(counts)
                if count!=len(amp): raise ValueError(f'{rid}: pulses outside rendered phase range')
                scenarios=['clean'] if db is None else [f'val_awgn_{target}_snr{db}dB' for target in [role,'both']]
                for scenario in scenarios:
                    q=read(args.output/'images'/scenario/'quality'/f'{rid}.json')
                    if counts!=[q['tile_peak_counts'][t['record_id']][role] for t in tiles]:
                        raise ValueError(f'{rid} {role} {scenario}: saved pulse counts mismatch')
                    if db is not None and abs(q['noise'][role]['measured_snr_db']-measured)>1e-6:
                        raise ValueError(f'{rid} {role} {scenario}: saved SNR mismatch')
                clipped=int((legacy_normalize_amplitude_global(amp,m['preprocessing']['a_ref'][role],mode='asinh',gamma=2.)>=1.).sum())
                if db is None: baseline=count
                rows.append(dict(class_name=source['class_name'],role=role,db=db,count=count,clipped=clipped,baseline=baseline))
            del signal,clean
        del waves,phase
        if i==1 or i%10==0 or i==len(grouped): print(f'Checked [{i}/{len(grouped)}] {rid}',flush=True)
    print('\nclass | channel | SNR | records | mean pulses | pooled retention % | pooled clipped % | median record clipped % | empty records | >=90% clipped records',flush=True)
    for cls in m['classes']:
        for role in ['uhf','hfct']:
            for db in levels:
                selected=[r for r in rows if r['class_name']==cls and r['role']==role and r['db']==db]
                if not selected: raise ValueError(f'Missing class {cls}')
                total=sum(r['count'] for r in selected); base=sum(r['baseline'] for r in selected)
                fractions=[r['clipped']/r['count'] for r in selected if r['count']]
                fmt=lambda x:'NA' if x is None else f'{x:.2f}'
                retained=100*total/base if base else None
                pooled=100*sum(r['clipped'] for r in selected)/total if total else None
                median=100*float(np.median(fractions)) if fractions else None
                label='clean' if db is None else f'{db}dB'
                print(f'{cls} | {role} | {label} | {len(selected)} | {total/len(selected):.1f} | '
                    f'{fmt(retained)} | {fmt(pooled)} | {fmt(median)} | '
                    f'{sum(r["count"]==0 for r in selected)} | {sum(f>=.9 for f in fractions)}',flush=True)
    print('Retention compares pulse COUNTS, not matched physical events; may exceed 100% due to spurious detections.',flush=True)
    print('Pooled clipping is pulse-weighted; record median excludes empty records, which are listed separately.',flush=True)
    print('All saved counts and measured SNRs matched. No files written.',flush=True)


def diagnose_snr(args):
    """Inspect one fixed validation parent; only write one requested visual comparison."""
    from PIL import Image, ImageDraw
    from generate_prpd_wfm import legacy_compute_phase_from_ch2,legacy_detect_pulse_peaks
    if args.results is None: raise ValueError('diagnose-snr requires --results (saved SNR evaluation directory)')
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    m=read(args.output/'manifest.json')
    if m.get('representation')!='legacy_phase_tiles_30deg': raise ValueError('Expected tile manifest')
    parents=sorted({r['parent_record_id'] for r in m['records'] if r['split']=='val'})
    if args.record_ids and (len(args.record_ids)!=1 or args.record_ids[0] not in parents):
        raise ValueError('Choose exactly one validation parent with --record-ids')
    rid=args.record_ids[0] if args.record_ids else parents[0]
    tiles=sorted([r for r in m['records'] if r['parent_record_id']==rid],key=lambda r:r['phase_start_deg'])
    if len(tiles)!=12 or any(t['split']!='val' for t in tiles): raise ValueError('Expected 12 validation tiles')
    source=dict(tiles[0],record_id=rid)
    roles=['uhf','hfct']
    scenarios=[('clean',None,None)]+[(f'val_awgn_{target}_snr{db}dB',target,db)
        for target in ['both','uhf','hfct'] for db in [20,10,0,-5]]
    print(f'Fixed validation record: {rid}; true class: {source["class_name"]}',flush=True)
    waves=read_parent_record(args.data,source); fs=source['sample_rate_hz']
    phase,_=legacy_compute_phase_from_ch2(waves['phase_ref'].values(),fs,smooth_win=21)
    stats={}
    print('channel | condition | independently measured SNR dB | pulses | clipped fraction',flush=True)
    for role in roles:
        clean=waves[role].values(); power=np.mean((clean-np.median(clean))**2)
        for db in [None,20,10,0,-5]:
            signal=clean; actual=None
            if db is not None:
                seed=int.from_bytes(hashlib.sha256(f'2026/awgn/{rid}/{role}'.encode()).digest()[:8],'little')
                signal,_=add_awgn_snr(clean,db,np.random.default_rng(seed))
                actual=float(10*np.log10(power/np.mean((signal-clean)**2)))
                if abs(actual-db)>1e-6: raise ValueError('Measured SNR mismatch')
            ph,amp=legacy_detect_pulse_peaks(signal,phase,fs,win_us=2.,thr_k=8.)[:2]
            counts=[int(((ph>=t['phase_start_deg'])&
                ((ph<t['phase_stop_deg']) if t['phase_stop_deg']<360 else (ph<=360))).sum()) for t in tiles]
            clipped=float(np.mean(amp>=m['preprocessing']['a_ref'][role])) if len(amp) else 0.
            stats[role,db]=(actual,counts,clipped)
            label='clean' if db is None else f'{db}dB'
            snr='NA' if actual is None else f'{actual:.6f}'
            print(f'{role} | {label} | {snr} | {sum(counts)} | {clipped:.2%}',flush=True)
    for scenario,target,db in scenarios:
        q=read(args.output/'images'/scenario/'quality'/f'{rid}.json')
        for role in roles:
            level=db if target in [role,'both'] else None
            actual,counts,_=stats[role,level]
            saved=[q['tile_peak_counts'][t['record_id']][role] for t in tiles]
            if saved!=counts: raise ValueError(f'{scenario} {role}: saved pulse count mismatch')
            if level is not None and abs(q['noise'][role]['measured_snr_db']-actual)>1e-6:
                raise ValueError(f'{scenario} {role}: saved SNR mismatch')
        print(f'QC {scenario}: PASS',flush=True)
    print('scenario | seed | mean UHF/HFCT weights | saturated U/H tile fraction | prediction',flush=True)
    tile_ids={t['record_id'] for t in tiles}
    for scenario,_,_ in scenarios:
        for seed in [42,43,44,45,46]:
            d=read(args.results/f'gfm_seed{seed}'/f'val_{scenario}.json')
            parent=next(p for p in d['predictions'] if p['record_id']==rid)
            rows=[p for p in d['tile_predictions'] if p['record_id'] in tile_ids]
            if len(rows)!=12: raise ValueError('Missing tile predictions')
            w=np.asarray([p['weights'] for p in rows]); mean=w.mean(axis=0); sat=(w>=.99).mean(axis=0)
            print(f'{scenario} | {seed} | {mean[0]:.4f}/{mean[1]:.4f} | '
                f'{sat[0]:.1%}/{sat[1]:.1%} | {m["classes"][parent["pred"]]}',flush=True)
    size=48; column=12*size+20; height=90
    canvas=Image.new('RGB',(column*2,55+height*len(scenarios)),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((10,5),f'Record: {rid}',fill='black')
    for j,role in enumerate(roles): draw.text((j*column+10,25),f'{role.upper()} | 12 blocks: 0-30 ... 330-360 deg',fill='black')
    for i,(scenario,target,db) in enumerate(scenarios):
        y=55+i*height
        for j,role in enumerate(roles):
            level=db if target in [role,'both'] else None
            draw.text((j*column+10,y),f'{scenario} | pulses={sum(stats[role,level][1])}',fill='black')
            for k,t in enumerate(tiles):
                path=args.output/'images'/scenario/role/t['class_name']/(t['record_id']+'.png')
                with Image.open(path) as image:
                    thumb=image.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
                x=j*column+10+k*size; canvas.paste(thumb,(x,y+20))
                draw.rectangle((x,y+20,x+size-1,y+size+19),outline='#bbbbbb')
    out=args.results/f'diagnostic_{rid}.png'
    canvas.save(out)
    print(f'PRPD comparison: {out}',flush=True)
    print('Done. No training, model changes, or result overwrites.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','render','tile-fit','tile-render','region-fit','region-resplit','region-render','diagnose-snr','audit-snr','audit-gates','audit-reliability'])
    p.add_argument('--split-seed',type=int,default=3407,help='Original-record split seed for audit or region-resplit; existing manifests are never edited')
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'revision_output')
    p.add_argument('--checksum',action='store_true')
    p.add_argument('--fraction',type=float,default=0.)
    p.add_argument('--snr-db',type=float,help='Tile-render AWGN measured waveform SNR in dB; exclusive with --fraction')
    p.add_argument('--train-bank',action='store_true',help='Render one deterministic random-SNR/target AWGN variant for every training parent only')
    p.add_argument('--target',choices=['both','uhf','hfct'],default='both')
    p.add_argument('--limit',type=int)
    p.add_argument('--smoke',action='store_true',help='Render two records per class per split for pipeline check')
    p.add_argument('--record-ids',nargs='+',help='Process exact manifest record IDs, e.g. to verify a CSV adapter')
    p.add_argument('--parent-manifest',type=Path,help='Original-record manifest; tile-fit inherits its split')
    p.add_argument('--results',type=Path,help='Saved SNR validation evaluation directory for diagnose-snr or audit-gates')
    p.add_argument('--noise-split',choices=['val','test'],default='test',help='Tile-noise split; validation scenarios use val_ prefix')
    p.add_argument('--render-splits',nargs='+',choices=['train','val','test'],default=['train','val'],help='Clean region views: default train and val only')
    a=p.parse_args()
    if a.train_bank and a.action!='tile-render': p.error('train-bank requires tile-render')
    if not np.isfinite(a.fraction) or a.fraction<0: p.error('fraction must be finite and nonnegative')
    if a.snr_db is not None and (not np.isfinite(a.snr_db) or not -100<=a.snr_db<=100 or a.fraction or a.action not in ['tile-render','region-render']):
        p.error('snr-db requires tile-render or region-render, finite -100..100 dB, and fraction 0')
    {'audit':audit,'render':render,'tile-fit':tile_fit,'tile-render':tile_render,'region-fit':region_fit,'region-resplit':region_resplit,'region-render':region_render,'diagnose-snr':diagnose_snr,'audit-snr':audit_snr,'audit-gates':audit_gates,'audit-reliability':audit_reliability}[a.action](a)
