"""Frozen whole-PRPD models, session-grouped validation records, 18 fixed raw-AWGN scenarios.

No training, selection, amplitude normalization, test-image access or manifest
rewrites. Preserve exact clean images for unaffected channels. Both/single use
identical corruptions. Report detected/visible peaks, actual white images/SNR.
"""
import argparse
import csv
import hashlib
import inspect
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import generate_prpd_wfm as g
from session_split import validate_grouped_manifest
from revision_data import read_parent_record,add_awgn_snr,image_paths

METHODS=['uhf','hfct','concat','equal','gfm','gfm_no_conf']
SNRS=[10,5,0,-15,-10,-5]
TARGETS=['uhf','hfct','both']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def scenario(target,db):return f'val_awgn_{target}_snr{db}dB'
def image_stats(path):
    with Image.open(path) as image:
        if image.size!=(224,224):raise ValueError('Expected 224x224 image: '+str(path))
        pixels=np.asarray(image.convert('RGB'))
    return dict(image_blank=bool(np.all(pixels==255)),ink_pixel_fraction=float(np.any(pixels<255,axis=2).mean()))
def peak_stats(amp,axis):
    outside=int(np.sum(amp>axis));visible=int(np.sum((amp>=0)&(amp<=axis)))
    return dict(detections=len(amp),outside_axis=outside,visible_peaks=visible,
        outside_pct=100*outside/len(amp) if len(amp) else 0.,no_detections=len(amp)==0,
        no_visible_peaks=visible==0,peak_max_v=float(amp.max()) if len(amp) else None,y_axis_max_v=axis)


def prepare(args,m,parents,axes,scenarios):
    for i,r in enumerate(parents,1):
        quality_paths=[args.data/'images'/sc/'quality'/(r['record_id']+'.json') for sc in scenarios]
        if args.resume and all(q.is_file() for q in quality_paths) and all(
            p.is_file() for sc in scenarios for p in image_paths(args.data,sc,r)):
            print(f'Noise [{i}/{len(parents)}] already complete: {r["record_id"]}',flush=True);continue
        waves=read_parent_record(args.raw_data,r)
        phase,_=g.legacy_compute_phase_from_ch2(waves['phase_ref'].values(),r['sample_rate_hz'],smooth_win=21)
        clean_stats={}
        for role,clean_path in zip(['uhf','hfct'],image_paths(args.data,'clean',r)):
            ph,amp,threshold=g.legacy_detect_pulse_peaks(waves[role].values(),phase,r['sample_rate_hz'],win_us=2.,thr_k=8.)
            clean_stats[role]=dict(**peak_stats(amp,axes[role]),**image_stats(clean_path),threshold_v=float(threshold))
            saved=json.loads((args.data/'images'/'clean'/'quality'/(r['record_id']+'.json')).read_text())
            if saved['channels'][role]['detections']!=len(amp):raise ValueError('Clean detector count mismatch: '+r['record_id'])
        for db in SNRS:
            noisy_stats={}
            for index,role in enumerate(['uhf','hfct']):
                seed=int.from_bytes(hashlib.sha256(f'2026/awgn/{r["record_id"]}/{role}'.encode()).digest()[:8],'little')
                signal,noise=add_awgn_snr(waves[role].values(),db,np.random.default_rng(seed))
                if abs(noise['measured_snr_db']-db)>1e-6:raise ValueError('Measured SNR mismatch')
                ph,amp,threshold=g.legacy_detect_pulse_peaks(signal,phase,r['sample_rate_hz'],win_us=2.,thr_k=8.)
                if not np.all(np.isfinite(amp)):raise ValueError('Nonfinite amplitudes')
                single_path=image_paths(args.data,scenario(role,db),r)[index]
                g.legacy_save_scatter_tile(single_path,ph,amp,xlim=(0.,360.),ylim=(0.,axes[role]),
                    s=8,alpha=.9,dpi=256,bg_white=True,resize_to=224)
                noisy_stats[role]=dict(**peak_stats(amp,axes[role]),**image_stats(single_path),
                    threshold_v=float(threshold),noise=noise)
                # Same bytes for single-channel corruption and both-channel corruption.
                both_path=image_paths(args.data,scenario('both',db),r)[index]
                both_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(single_path,both_path)
                del signal
            for target in TARGETS:
                sc=scenario(target,db);details={}
                for index,role in enumerate(['uhf','hfct']):
                    corrupted=target in [role,'both']
                    if not corrupted:
                        destination=image_paths(args.data,sc,r)[index]
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(image_paths(args.data,'clean',r)[index],destination)
                    details[role]=dict(noisy_stats[role] if corrupted else clean_stats[role],corrupted=corrupted)
                dump(args.data/'images'/sc/'quality'/(r['record_id']+'.json'),
                    dict(record_id=r['record_id'],split='val',scenario=sc,channels=details))
        print(f'Noise [{i}/{len(parents)}] {r["record_id"]}: all 18 fixed scenarios complete',flush=True)
        del waves,phase


def main(args):
    raw=(args.data/'manifest.json').read_bytes();m=json.loads(raw);protocol=json.loads((args.source/'protocol.json').read_text())
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Training/data manifest mismatch')
    if m.get('representation')!='whole_record_prpd' or any('parent_record_id' in r for r in m['records']):raise ValueError('Expected complete-record inputs')
    pp=m['preprocessing']
    if pp.get('amplitude_normalization') is not False or pp.get('amplitude_clipping') is not False:raise ValueError('Expected raw volts without amplitude mapping')
    if pp.get('y_axis_fit_split')!='train':raise ValueError('Axes must be TRAIN-fitted')
    split_counts=validate_grouped_manifest(m)
    if 'armed' not in inspect.getsource(g.legacy_compute_phase_from_ch2):raise ValueError('Corrected CH2 entry missing')
    expected=m.get('amplitude_revision',{}).get('phase_code_sha256')
    if expected!=digest(Path(g.__file__)):raise ValueError('Phase/detector/renderer code changed since image generation')
    parents=[r for r in m['records'] if r['split']=='val'];axes=pp['y_axis_max_v']
    if any(not np.isfinite(axes[k]) or axes[k]<=0 for k in ['uhf','hfct']):raise ValueError('Invalid fixed axes')
    for r in parents:
        for p in image_paths(args.data,'clean',r):
            if not p.is_file():raise FileNotFoundError(p)
    checkpoints={method:args.source/f'{method}_seed3407'/'best.pt' for method in METHODS}
    checkpoint_hashes={method:digest(p) for method,p in checkpoints.items()}
    scenarios=[scenario(target,db) for db in SNRS for target in TARGETS]
    tracked=[Path(__file__),ROOT/'revision_data.py',ROOT/'revision_train.py',ROOT/'generate_prpd_wfm.py',ROOT/'mso44_wfm.py',ROOT/'mso44_csv.py']
    settings=dict(source_manifest_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=checkpoint_hashes,
        code_sha256={p.name:digest(p) for p in tracked},source=str(args.source),data=str(args.data),raw_data=str(args.raw_data),
        seed=3407,noise_seed=2026,snr_db=SNRS,targets=TARGETS,records=split_counts['val'],evaluation_split='val',
        y_axis_max_v=axes,amplitude_normalization=False,metric_unit='original_record',batch_size=32,
        snr_definition='10log10(mean((clean-median(clean))**2)/mean(added_noise**2)); entire captured waveform including existing background',
        pairing='Same standardized Gaussian realization per parent/channel across SNRs; both/single copies identical noisy images; CH2 never corrupted',
        selection='Frozen best checkpoints; no optimization, no noise-based checkpoint selection, no test images')
    info=args.output/'protocol.json'
    if args.resume:
        if not info.is_file() or json.loads(info.read_text())!=settings:raise ValueError('Resume protocol mismatch')
    else:
        if args.output.exists() or any((args.data/'images'/sc).exists() for sc in scenarios):
            raise FileExistsError('Output/scenarios already exist. Use --resume only for this exact protocol.')
        args.output.mkdir(parents=True,exist_ok=False);dump(info,settings)
    for sc in scenarios:
        config=args.data/'images'/sc/'config.json';value=dict(split='val',scenario=sc,protocol=settings)
        if config.exists() and json.loads(config.read_text())!=value:raise ValueError('Scenario config mismatch')
        dump(config,value)
    prepare(args,m,parents,axes,scenarios)
    quality_rows=[]
    for sc in scenarios:
        for r in parents:
            q=json.loads((args.data/'images'/sc/'quality'/(r['record_id']+'.json')).read_text())
            for role in ['uhf','hfct']:
                d=q['channels'][role]
                quality_rows.append(dict(scenario=sc,record_id=r['record_id'],class_name=r['class_name'],channel=role,
                    corrupted=d['corrupted'],detections=d['detections'],visible_peaks=d['visible_peaks'],outside_axis=d['outside_axis'],
                    outside_pct=d['outside_pct'],image_blank=d['image_blank'],no_visible_peaks=d['no_visible_peaks'],
                    ink_pixel_fraction=d['ink_pixel_fraction'],measured_snr_db=d.get('noise',{}).get('measured_snr_db')))
    with (args.output/'image_quality.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(quality_rows[0]));w.writeheader();w.writerows(quality_rows)
    for sc in scenarios:
        for role in ['uhf','hfct']:
            rows=[r for r in quality_rows if r['scenario']==sc and r['channel']==role]
            print('QUALITY',sc,role,'outside_records=',sum(r['outside_axis']>0 for r in rows),
                'white_images=',sum(r['image_blank'] for r in rows),flush=True)
    if any(digest(p)!=checkpoint_hashes[method] for method,p in checkpoints.items()):raise ValueError('Checkpoint changed during preparation')
    import torch
    from torch.utils.data import DataLoader
    from revision_train import Paired,build_model,evaluate,AUX_METHODS,summarize
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; images ready, rerun --resume after correcting environment')
    device=torch.device('cuda');print('Frozen evaluation device:',device,flush=True)
    results=[];predictions={}
    for method in METHODS:
        checkpoint=torch.load(checkpoints[method],map_location='cpu',weights_only=True)
        if checkpoint['method']!=method or checkpoint['seed']!=3407 or checkpoint['classes']!=m['classes']:raise ValueError('Checkpoint identity mismatch')
        model=build_model(method,checkpoint['backbone'],checkpoint['feat_dim'],len(m['classes']),False).to(device)
        model.load_state_dict(checkpoint['state_dict']);model.requires_grad_(False);model.eval()
        for sc in ['clean']+scenarios:
            metrics=evaluate(model,DataLoader(Paired(m,args.data,'val',sc),batch_size=32,shuffle=False),device,m['classes'],diagnostics=method in AUX_METHODS)
            if metrics['n']!=split_counts['val']:raise ValueError('Unexpected evaluation count')
            metrics.update(evaluation_split='val',metric_unit='original_record',selection_used_this_set=False)
            if sc=='clean':
                old=json.loads((args.source/f'{method}_seed3407'/'val_clean.json').read_text())
                if metrics['correct']!=old['correct'] or abs(metrics['macro_f1']-old['macro_f1'])>1e-10:raise ValueError('Clean reproduction failed')
            preds={p['record_id']:p['pred'] for p in metrics['predictions']}
            if sc=='clean':predictions[method]=preds
            elif (method=='uhf' and '_hfct_' in sc) or (method=='hfct' and '_uhf_' in sc):
                if preds!=predictions[method]:raise ValueError('Unaffected single-channel predictions changed')
            dump(args.output/f'{method}_seed3407'/f'val_{sc}.json',metrics)
            results.append(dict(method=method,seed=3407,scenario=sc,evaluation_split='val',metric_unit='original_record',
                **{k:metrics[k] for k in ['n','correct','accuracy','macro_f1','balanced_accuracy']},macro_precision=metrics['report']['macro avg']['precision']))
            dump(args.output/'results.json',results)
            print(f'{method} seed=3407 {sc} n={split_counts["val"]} acc={metrics["accuracy"]:.4f} f1={metrics["macro_f1"]:.4f}',flush=True)
        del model,checkpoint;torch.cuda.empty_cache()
    if len(results)!=114:raise ValueError('Expected 114 results')
    if any(digest(p)!=checkpoint_hashes[method] for method,p in checkpoints.items()) or (args.data/'manifest.json').read_bytes()!=raw:
        raise ValueError('Source checkpoint/manifest changed')
    summarize(args.output,results)
    with (args.output/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print('DONE: 114 frozen-model validation results; 18 noise scenarios + clean; all record counts audited. Checkpoint hashes unchanged.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--raw-data',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    main(p.parse_args())

