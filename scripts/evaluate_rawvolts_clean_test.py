"""Fixed clean test evaluation, 158 records; no training or model selection.

Eight single-sensor baselines + four dual-sensor fusion variants. Sources fixed
before evaluating: original six ResNet models, CNN/SVM study, low-LR Swin retry.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import joblib
from torch.utils.data import DataLoader
from revision_train import Paired,build_model,evaluate,AUX_METHODS
from rawvolts_single_baselines import SensorCNN,metrics
from baseline_features import load_features


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def csv_save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(args):
    raw=(args.data/'manifest.json').read_bytes();m=json.loads(raw);manifest_hash=hashlib.sha256(raw).hexdigest()
    if args.output.exists():raise FileExistsError('Use a fresh output directory; no overwrite/resume')
    if m.get('representation')!='whole_record_prpd' or m['preprocessing'].get('amplitude_normalization') is not False:
        raise ValueError('Expected complete raw-voltage PRPD')
    if len({r['record_id'] for r in m['records']})!=786 or len({r['group_id'] for r in m['records']})!=786:raise ValueError('Duplicate record/group IDs')
    if dict(Counter(r['split'] for r in m['records']))!={'train':471,'val':157,'test':158}:raise ValueError('Unexpected split')
    for folder in [args.fusion_source,args.baseline_source,args.swin_source]:
        protocol=json.loads((folder/'protocol.json').read_text())
        if protocol['manifest_sha256']!=manifest_hash:raise ValueError('Source manifest mismatch: '+str(folder))
    plans=[]
    for role in ['uhf','hfct']:
        plans.extend([
            dict(key='svm_'+role,method='svm',modality=role,path=args.baseline_source/f'svm_{role}'/'model.joblib'),
            dict(key='cnn_'+role,method='cnn',modality=role,path=args.baseline_source/f'cnn_{role}'/'best.pt'),
            dict(key='resnet50_'+role,method='resnet50',modality=role,path=args.fusion_source/f'{role}_seed3407'/'best.pt'),
            dict(key='swin_t_'+role,method='swin_t',modality=role,path=args.swin_source/f'swin_t_{role}'/'best.pt')])
    for method in ['concat','equal','gfm','gfm_no_conf']:
        plans.append(dict(key=method,method=method,modality='dual',path=args.fusion_source/f'{method}_seed3407'/'best.pt'))
    hashes={str(p['path']):digest(p['path']) for p in plans}
    dataset=Paired(m,args.data,'test');loader=DataLoader(dataset,batch_size=32,shuffle=False)
    if len(dataset)!=158:raise ValueError('Expected 158 test records')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no test inference started')
    torch.set_num_threads(4);device=torch.device('cuda')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',dict(manifest_sha256=manifest_hash,
        arguments={k:str(v) for k,v in vars(args).items()},source_checkpoint_sha256=hashes,
        script_sha256=digest(Path(__file__)),evaluation_split='test',n=158,seed=3407,
        selection='Fixed source best checkpoints, selected on clean val; Swin retry lr1e-5; no training/test-driven selection',
        averaging='Recall and F1 are unweighted class means (macro); pixel preprocessing unchanged'))
    rows=[]
    for plan in plans:
        method,role,path=plan['method'],plan['modality'],plan['path']
        print('Evaluating frozen:',plan['key'],path,flush=True)
        if method=='svm':
            bundle=joblib.load(path)
            if bundle['manifest_sha256']!=manifest_hash or bundle['classes']!=m['classes'] or bundle['modality']!=role:
                raise ValueError('SVM identity mismatch')
            x,y=load_features(dataset.records,args.data,role,64,m['classes'])
            predictions=bundle['svm'].predict(bundle['scaler'].transform(bundle['pca'].transform(x)))
            d=dict(predictions=[dict(record_id=r['record_id'],true=int(t),pred=int(p)) for r,t,p in zip(dataset.records,y,predictions)])
            del bundle,x,y,predictions
        else:
            c=torch.load(path,map_location='cpu',weights_only=True)
            if c['classes']!=m['classes'] or c['seed']!=3407:raise ValueError('Checkpoint classes/seed mismatch')
            if method in ['cnn','swin_t']:
                if c['method']!=method or c['modality']!=role or c['manifest_sha256']!=manifest_hash:
                    raise ValueError('Baseline identity mismatch')
                model=SensorCNN(role,len(m['classes'])) if method=='cnn' else build_model('swin_'+role,'resnet50',512,len(m['classes']),False)
            else:
                expected=role if method=='resnet50' else method
                if c['method']!=expected:raise ValueError('Fusion identity mismatch')
                model=build_model(expected,c['backbone'],c['feat_dim'],len(m['classes']),False)
            model.load_state_dict(c['state_dict']);model.to(device).requires_grad_(False);model.eval()
            d=evaluate(model,loader,device,m['classes'],diagnostics=method in AUX_METHODS)
            d['selected_epoch']=c['best_epoch']
            del model,c;torch.cuda.empty_cache()
        result=metrics(d['predictions'],m['classes'])
        if result['n']!=158:raise ValueError('Unexpected test prediction count')
        result.update(evaluation_split='test',metric_unit='original_record',checkpoint_sha256=hashes[str(path)])
        for key in ['branch_accuracy','gate','selected_epoch']:
            if key in d:result[key]=d[key]
        save(args.output/plan['key']/'test_clean.json',result)
        row=dict(method=method,modality=role,n=158,evaluation_split='test',
            **{k:result[k] for k in ['accuracy','recall_mean','f1_mean']})
        rows.append(row);save(args.output/'results.json',rows)
        print(f'TEST {method}/{role}: n=158 Acc={100*row["accuracy"]:.2f} Recall(mean)={100*row["recall_mean"]:.2f} F1(mean)={100*row["f1_mean"]:.2f}',flush=True)
    if len(rows)!=12 or any(digest(Path(p))!=h for p,h in hashes.items()) or (args.data/'manifest.json').read_bytes()!=raw:
        raise ValueError('Source files changed or incomplete results')
    main_table=[r for r in rows if r['modality']!='dual' or r['method']=='gfm']
    ablation=[r for r in rows if r['modality']=='dual']
    csv_save(args.output/'all_clean_test.csv',rows);csv_save(args.output/'main_table.csv',main_table);csv_save(args.output/'ablation_table.csv',ablation)
    print('DONE: 12 clean TEST results, all n=158; main table9 rows, ablation4 rows. Checkpoints unchanged; no training.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','fusion-source','baseline-source','swin-source','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
