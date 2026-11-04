"""Frozen 12-model test: 158 original records, clean + 18 prespecified AWGN cases.

Use the exact checkpoints already used for clean test. No training/selection,
normalization, axis refit or changes to clean images/manifest/checkpoints.
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
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader
import generate_prpd_wfm as g
from revision_data import read_parent_record,add_awgn_snr,image_paths
from full360_fixed_noise_val import digest,dump,image_stats,peak_stats
from evaluate_rawvolts_clean_test import Paired,build_model,evaluate,AUX_METHODS,SensorCNN,metrics,load_features

SNRS=[10,5,0,-15,-10,-5]
TARGETS=['uhf','hfct','both']
def scenario(target,db):return f'test_awgn_{target}_snr{db}dB'
def csv_save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plans_from_clean(protocol):
    a=protocol['arguments'];fusion=Path(a['fusion_source']);base=Path(a['baseline_source']);swin=Path(a['swin_source'])
    plans=[]
    for role in ['uhf','hfct']:
        plans.extend([
            dict(key='svm_'+role,method='svm',modality=role,path=base/f'svm_{role}'/'model.joblib'),
            dict(key='cnn_'+role,method='cnn',modality=role,path=base/f'cnn_{role}'/'best.pt'),
            dict(key='resnet50_'+role,method='resnet50',modality=role,path=fusion/f'{role}_seed3407'/'best.pt'),
            dict(key='swin_t_'+role,method='swin_t',modality=role,path=swin/f'swin_t_{role}'/'best.pt')])
    for method in ['concat','equal','gfm','gfm_no_conf']:
        plans.append(dict(key=method,method=method,modality='dual',path=fusion/f'{method}_seed3407'/'best.pt'))
    if {str(p['path']) for p in plans}!=set(protocol['source_checkpoint_sha256']):raise ValueError('Checkpoint plans differ from clean test')
    return plans


def prepare(args,parents,axes,scenarios):
    for i,r in enumerate(parents,1):
        quality_paths=[args.data/'images'/sc/'quality'/(r['record_id']+'.json') for sc in scenarios]
        if args.resume and all(q.is_file() for q in quality_paths) and all(p.is_file() for sc in scenarios for p in image_paths(args.data,sc,r)):
            print(f'Noise TEST [{i}/158] already complete: {r["record_id"]}',flush=True);continue
        waves=read_parent_record(args.raw_data,r)
        if not np.isclose(waves['phase_ref'].info.sample_rate_hz,r['sample_rate_hz'],rtol=1e-10,atol=0):raise ValueError('CH2 sampling rate mismatch')
        phase,_=g.legacy_compute_phase_from_ch2(waves['phase_ref'].values(),r['sample_rate_hz'],smooth_win=21)
        clean_stats={}
        saved=json.loads((args.data/'images'/'clean'/'quality'/(r['record_id']+'.json')).read_text())
        for role,clean_path in zip(['uhf','hfct'],image_paths(args.data,'clean',r)):
            ph,amp,threshold=g.legacy_detect_pulse_peaks(waves[role].values(),phase,r['sample_rate_hz'],win_us=2.,thr_k=8.)
            if saved['channels'][role]['detections']!=len(amp):raise ValueError('Clean detector count mismatch')
            clean_stats[role]=dict(**peak_stats(amp,axes[role]),**image_stats(clean_path),threshold_v=float(threshold))
        for db in SNRS:
            noisy_stats={}
            for index,role in enumerate(['uhf','hfct']):
                seed=int.from_bytes(hashlib.sha256(f'2026/awgn/{r["record_id"]}/{role}'.encode()).digest()[:8],'little')
                signal,noise=add_awgn_snr(waves[role].values(),db,np.random.default_rng(seed))
                if abs(noise['measured_snr_db']-db)>1e-6:raise ValueError('Measured SNR mismatch')
                ph,amp,threshold=g.legacy_detect_pulse_peaks(signal,phase,r['sample_rate_hz'],win_us=2.,thr_k=8.)
                if not np.all(np.isfinite(amp)):raise ValueError('Nonfinite amplitudes')
                single=image_paths(args.data,scenario(role,db),r)[index]
                g.legacy_save_scatter_tile(single,ph,amp,xlim=(0.,360.),ylim=(0.,axes[role]),
                    s=8,alpha=.9,dpi=256,bg_white=True,resize_to=224)
                noisy_stats[role]=dict(**peak_stats(amp,axes[role]),**image_stats(single),threshold_v=float(threshold),noise=noise)
                both=image_paths(args.data,scenario('both',db),r)[index]
                both.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(single,both);del signal
            for target in TARGETS:
                sc=scenario(target,db);details={}
                for index,role in enumerate(['uhf','hfct']):
                    corrupted=target in [role,'both']
                    if not corrupted:
                        destination=image_paths(args.data,sc,r)[index];destination.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(image_paths(args.data,'clean',r)[index],destination)
                    details[role]=dict(noisy_stats[role] if corrupted else clean_stats[role],corrupted=corrupted)
                dump(args.data/'images'/sc/'quality'/(r['record_id']+'.json'),dict(record_id=r['record_id'],split='test',scenario=sc,channels=details))
        print(f'Noise TEST [{i}/158] {r["record_id"]}: all 18 scenarios complete',flush=True)
        del waves,phase


def main(args):
    raw=(args.data/'manifest.json').read_bytes();m=json.loads(raw);manifest_hash=hashlib.sha256(raw).hexdigest()
    clean_protocol=json.loads((args.clean_source/'protocol.json').read_text());val_protocol=json.loads((args.val_noise_source/'protocol.json').read_text())
    if clean_protocol['manifest_sha256']!=manifest_hash or val_protocol['source_manifest_sha256']!=manifest_hash:raise ValueError('Manifest differs from frozen clean/validation studies')
    if m.get('representation')!='whole_record_prpd' or any('parent_record_id' in r for r in m['records']):raise ValueError('Expected full360 original records')
    if len({r['record_id'] for r in m['records']})!=786 or len({r['group_id'] for r in m['records']})!=786:raise ValueError('Duplicate records/groups')
    if dict(Counter(r['split'] for r in m['records']))!={'train':471,'val':157,'test':158}:raise ValueError('Split mismatch')
    pp=m['preprocessing'];axes=pp['y_axis_max_v']
    if pp.get('amplitude_normalization') is not False or pp.get('amplitude_clipping') is not False or pp.get('y_axis_fit_split')!='train':raise ValueError('Expected raw volts and frozen TRAIN axes')
    if any(not np.isfinite(axes[k]) or axes[k]<=0 for k in ['uhf','hfct']):raise ValueError('Invalid axes')
    if axes!=val_protocol['y_axis_max_v'] or val_protocol['snr_db']!=SNRS or val_protocol['noise_seed']!=2026:raise ValueError('Noise/axis protocol differs from validation')
    if 'armed' not in inspect.getsource(g.legacy_compute_phase_from_ch2) or digest(Path(g.__file__))!=m['amplitude_revision']['phase_code_sha256']:raise ValueError('Phase/detector/renderer changed')
    plans=plans_from_clean(clean_protocol);hashes=clean_protocol['source_checkpoint_sha256']
    if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Checkpoint changed since clean test')
    for plan in plans:
        golden=args.clean_source/plan['key']/'test_clean.json'
        if not golden.is_file():raise FileNotFoundError(golden)
    parents=[r for r in m['records'] if r['split']=='test']
    for r in parents:
        for p in image_paths(args.data,'clean',r):
            if not p.is_file():raise FileNotFoundError(p)
    scenarios=[scenario(target,db) for db in SNRS for target in TARGETS]
    tracked=[Path(__file__),ROOT/'generate_prpd_wfm.py',ROOT/'revision_data.py',ROOT/'revision_train.py',
        Path(__file__).with_name('full360_fixed_noise_val.py'),Path(__file__).with_name('evaluate_rawvolts_clean_test.py'),
        Path(__file__).with_name('rawvolts_single_baselines.py'),Path(__file__).with_name('baseline_features.py'),ROOT/'recovered_models.py',ROOT/'mso44_wfm.py',ROOT/'mso44_csv.py']
    settings=dict(manifest_sha256=manifest_hash,checkpoint_sha256=hashes,clean_protocol_sha256=digest(args.clean_source/'protocol.json'),
        code_sha256={p.name:digest(p) for p in tracked},data=str(args.data),raw_data=str(args.raw_data),clean_source=str(args.clean_source),
        val_noise_source=str(args.val_noise_source),evaluation_split='test',n=158,seed=3407,noise_seed=2026,snr_db=SNRS,targets=TARGETS,
        y_axis_max_v=axes,selection='Exact clean-test checkpoints; no training, selection or axis refit',
        snr_definition=val_protocol['snr_definition'],pairing=val_protocol['pairing'],metric_averaging='Class-equal mean Recall/F1 (macro)')
    info=args.output/'protocol.json'
    if args.resume:
        if not info.is_file() or json.loads(info.read_text())!=settings:raise ValueError('Resume protocol mismatch')
    else:
        if args.output.exists() or any((args.data/'images'/sc).exists() for sc in scenarios):raise FileExistsError('Output/scenarios exist; use --resume for this exact protocol only')
        args.output.mkdir(parents=True,exist_ok=False);dump(info,settings)
    for sc in scenarios:
        p=args.data/'images'/sc/'config.json';value=dict(split='test',scenario=sc,protocol=settings)
        if p.exists() and json.loads(p.read_text())!=value:raise ValueError('Scenario config mismatch')
        dump(p,value)
    prepare(args,parents,axes,scenarios)
    quality=[]
    for sc in scenarios:
        for r in parents:
            q=json.loads((args.data/'images'/sc/'quality'/(r['record_id']+'.json')).read_text())
            for role in ['uhf','hfct']:
                d=q['channels'][role]
                quality.append(dict(scenario=sc,record_id=r['record_id'],class_name=r['class_name'],channel=role,corrupted=d['corrupted'],
                    detections=d['detections'],outside_axis=d['outside_axis'],outside_pct=d['outside_pct'],visible_peaks=d['visible_peaks'],
                    image_blank=d['image_blank'],no_visible_peaks=d['no_visible_peaks'],ink_pixel_fraction=d['ink_pixel_fraction'],
                    measured_snr_db=d.get('noise',{}).get('measured_snr_db')))
    csv_save(args.output/'image_quality.csv',quality)
    if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Checkpoint changed during rendering')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; images ready, resume after correcting environment')
    torch.set_num_threads(4);device=torch.device('cuda');rows=[]
    for plan in plans:
        method,role,path=plan['method'],plan['modality'],plan['path'];reference=None
        if method=='svm':
            bundle=joblib.load(path)
            if bundle['manifest_sha256']!=manifest_hash or bundle['classes']!=m['classes'] or bundle['modality']!=role:raise ValueError('SVM identity mismatch')
        else:
            c=torch.load(path,map_location='cpu',weights_only=True)
            if c['classes']!=m['classes'] or c['seed']!=3407:raise ValueError('Checkpoint identity mismatch')
            if method in ['cnn','swin_t']:
                if c['method']!=method or c['modality']!=role or c['manifest_sha256']!=manifest_hash:raise ValueError('Baseline identity mismatch')
                model=SensorCNN(role,len(m['classes'])) if method=='cnn' else build_model('swin_'+role,'resnet50',512,len(m['classes']),False)
            else:
                expected=role if method=='resnet50' else method
                if c['method']!=expected:raise ValueError('Fusion identity mismatch')
                model=build_model(expected,c['backbone'],c['feat_dim'],len(m['classes']),False)
            model.load_state_dict(c['state_dict']);model.to(device).requires_grad_(False);model.eval()
        for sc in ['clean']+scenarios:
            dataset=Paired(m,args.data,'test',sc)
            if method=='svm':
                # Same 64px grayscale/PCA/scaler as fitted on train; no refitting.
                # Loader helper is clean-only, so read the requested scenario explicitly.
                from PIL import Image
                x=np.empty((158,64*64),dtype=np.float32);truth=[]
                for k,r in enumerate(dataset.records):
                    p=image_paths(args.data,sc,r)[0 if role=='uhf' else 1]
                    with Image.open(p) as im:x[k]=np.asarray(im.convert('L').resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255.
                    truth.append(m['classes'].index(r['class_name']))
                predictions=bundle['svm'].predict(bundle['scaler'].transform(bundle['pca'].transform(x)))
                d=dict(predictions=[dict(record_id=r['record_id'],true=int(t),pred=int(p)) for r,t,p in zip(dataset.records,truth,predictions)]);del x
            else:
                d=evaluate(model,DataLoader(dataset,batch_size=32,shuffle=False),device,m['classes'],diagnostics=method in AUX_METHODS)
            result=metrics(d['predictions'],m['classes'])
            if result['n']!=158:raise ValueError('Unexpected evaluation count')
            pred={p['record_id']:(p['true'],p['pred']) for p in result['predictions']}
            if sc=='clean':
                golden=json.loads((args.clean_source/plan['key']/'test_clean.json').read_text())
                old={p['record_id']:(p['true'],p['pred']) for p in golden['predictions']}
                if pred!=old:raise ValueError('Clean predictions differ from original frozen test: '+plan['key'])
                reference=pred
            elif role in ['uhf','hfct'] and (('_hfct_' in sc and role=='uhf') or ('_uhf_' in sc and role=='hfct')):
                if pred!=reference:raise ValueError('Unaffected single-sensor predictions changed')
            result.update(evaluation_split='test',metric_unit='original_record',checkpoint_sha256=hashes[str(path)])
            for k in ['branch_accuracy','gate']:
                if k in d:result[k]=d[k]
            dump(args.output/plan['key']/f'test_{sc}.json',result)
            rows.append(dict(method=method,modality=role,scenario=sc,seed=3407,n=158,evaluation_split='test',
                **{k:result[k] for k in ['accuracy','recall_mean','f1_mean']}))
            dump(args.output/'results.json',rows)
            print(f'TEST {plan["key"]} {sc}: n=158 Acc={100*result["accuracy"]:.2f} Recall={100*result["recall_mean"]:.2f} F1={100*result["f1_mean"]:.2f}',flush=True)
        if method=='svm':del bundle
        else:del model,c;torch.cuda.empty_cache()
    if len(rows)!=228 or any(digest(Path(p))!=h for p,h in hashes.items()) or (args.data/'manifest.json').read_bytes()!=raw:
        raise ValueError('Incomplete results or source changed')
    csv_save(args.output/'all_test_results.csv',rows)
    csv_save(args.output/'main_table_by_snr.csv',[r for r in rows if r['modality']!='dual' or r['method']=='gfm'])
    csv_save(args.output/'fusion_ablation_by_snr.csv',[r for r in rows if r['modality']=='dual'])
    print('DONE: 228 frozen TEST results; 12 models x (clean +18 noise), all n=158. Clean predictions reproduced; checkpoint hashes unchanged; no training.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','raw-data','clean-source','val-noise-source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    main(p.parse_args())
