"""Matched clean single-sensor baselines for the 786-record raw-voltage dataset.

Train SVM/CNN/Swin-T for each sensor. Reuse independently trained ResNet50
and dual GFM results from the exact same manifest. Validation only; no test
images, resplitting, raw-waveform access or changes to existing checkpoints.
Report Accuracy, class-equal average Recall and class-equal average F1.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report
import joblib
from revision_train import Paired,CNNEncoder,evaluate,build_model
from baseline_features import load_features


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def metrics(predictions,classes):
    truth=[p['true'] for p in predictions];pred=[p['pred'] for p in predictions];labels=list(range(len(classes)))
    return dict(n=len(truth),accuracy=float(accuracy_score(truth,pred)),
        recall_mean=float(recall_score(truth,pred,labels=labels,average='macro',zero_division=0)),
        f1_mean=float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0)),
        averaging='unweighted arithmetic mean over the four classes (macro)',
        per_class=classification_report(truth,pred,labels=labels,target_names=classes,output_dict=True,zero_division=0),
        predictions=predictions)


class SensorCNN(nn.Module):
    def __init__(self,role,classes):
        super().__init__();self.role=role
        self.encoder=CNNEncoder('simple_cnn',512,classes,False)
    def forward(self,u,h):
        _,logits=self.encoder(u if self.role=='uhf' else h)
        return logits,None


def main(args):
    path=args.data/'manifest.json';raw=path.read_bytes();m=json.loads(raw)
    if args.output.exists():raise FileExistsError('Use a fresh output directory')
    if m.get('representation')!='whole_record_prpd' or any('parent_record_id' in r for r in m['records']):raise ValueError('Expected whole-record data')
    if len({r['record_id'] for r in m['records']})!=786 or len({r['group_id'] for r in m['records']})!=786:raise ValueError('Expected 786 unique groups')
    if dict(Counter(r['split'] for r in m['records']))!={'train':471,'val':157,'test':158}:raise ValueError('Unexpected split counts')
    if m['preprocessing'].get('amplitude_normalization') is not False:raise ValueError('Expected raw-voltage images')
    protocol=json.loads((args.source/'protocol.json').read_text())
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Existing model manifest mismatch')
    if protocol.get('backbone')!='resnet50' or protocol['arguments']['seeds']!=[3407]:raise ValueError('Expected existing ResNet50 seed3407 study')
    if args.noise_results:
        noise_protocol=json.loads((args.noise_results/'protocol.json').read_text())
        if noise_protocol['source_manifest_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Noise results/data mismatch')
    origins={method:args.source/f'{method}_seed3407'/'val_clean.json' for method in ['uhf','hfct','gfm']}
    hashes={str(p):digest(p) for p in origins.values()}
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),seed=args.seed,
        script_sha256=digest(Path(__file__)),source_results=hashes,epochs=args.epochs,batch_size=32,
        split_seed=m.get('split_seed'),evaluation_split='val',
        deep_input='One sensor RGB224, ImageNet pixel normalization; same saved PRPD as GFM; no augmentation/no noise',
        cnn='CNNEncoder: 3 convolution blocks32/64/128 + GAP; 128->512->ReLU->Dropout0.5->4; random initialization',
        swin='Existing recovered Swin-T single-sensor model, ImageNet initialization; 768->512->ReLU->Dropout0.5->4',
        svm='Single-sensor grayscale64 flatten/255; PCA128 TRAIN only; StandardScaler TRAIN only; RBF SVC C10 gamma=scale; no search',
        optimizer='CNN Adam lr0.001; Swin-T Adam lr0.00001; StepLR step10 gamma0.1; cross entropy',
        selection='Clean validation class-equal mean F1, first maximum; no test images'))
    table=[]
    def record(method,role,d):
        result=metrics(d['predictions'],m['classes'])
        if result['n']!=157:raise ValueError('Expected 157 val records')
        save(args.output/f'{method}_{role}'/'val_clean.json',result)
        table.append(dict(method=method,modality=role,evaluation_split='val',n=157,seed=args.seed,
            **{k:result[k] for k in ['accuracy','recall_mean','f1_mean']}))
        with (args.output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        print(f'VAL {method}/{role}: n=157 Acc={100*result["accuracy"]:.2f} Recall(mean)={100*result["recall_mean"]:.2f} F1(mean)={100*result["f1_mean"]:.2f}',flush=True)
    for method,role in [('uhf','uhf'),('hfct','hfct'),('gfm','dual')]:
        record('resnet50' if method!='gfm' else 'gfm',role,json.loads(origins[method].read_text()))
    train=[r for r in m['records'] if r['split']=='train'];val=[r for r in m['records'] if r['split']=='val']
    for role in ['uhf','hfct']:
        seed_all(args.seed)
        x,y=load_features(train,args.data,role,64,m['classes'])
        pca=PCA(n_components=128,svd_solver='randomized',random_state=args.seed)
        scaler=StandardScaler();z=scaler.fit_transform(pca.fit_transform(x))
        svm=SVC(C=10.,kernel='rbf',gamma='scale',decision_function_shape='ovr',random_state=args.seed)
        svm.fit(z,y)
        xv,yv=load_features(val,args.data,role,64,m['classes']);pred=svm.predict(scaler.transform(pca.transform(xv)))
        predictions=[dict(record_id=r['record_id'],true=int(t),pred=int(p)) for r,t,p in zip(val,yv,pred)]
        record('svm',role,dict(predictions=predictions))
        joblib.dump(dict(pca=pca,scaler=scaler,svm=svm,modality=role,classes=m['classes'],
            manifest_sha256=hashlib.sha256(raw).hexdigest()),args.output/f'svm_{role}'/'model.joblib')
        del x,y,z,xv,yv,pred,pca,scaler,svm
    if not torch.cuda.is_available():raise RuntimeError('SVM completed, but CUDA unavailable for remaining CNN/Swin training')
    device=torch.device('cuda');torch.set_num_threads(4)
    for name in ['cnn','swin_t']:
        for role in ['uhf','hfct']:
            seed_all(args.seed)
            loaders={split:DataLoader(Paired(m,args.data,split),batch_size=32,shuffle=split=='train',
                generator=torch.Generator().manual_seed(args.seed)) for split in ['train','val']}
            model=(SensorCNN(role,len(m['classes'])) if name=='cnn' else
                build_model('swin_'+role,'resnet50',512,len(m['classes']),True)).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5 if name=='swin_t' else .001)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
            best=-1.;folder=args.output/f'{name}_{role}';history=[]
            for epoch in range(1,args.epochs+1):
                model.train();total=0;loss_sum=0.
                for u,h,y,_ in loaders['train']:
                    optimizer.zero_grad(set_to_none=True)
                    logits,_=model(u.to(device),h.to(device));loss=nn.functional.cross_entropy(logits,y.to(device))
                    loss.backward();optimizer.step();total+=len(y);loss_sum+=float(loss.detach())*len(y)
                d=evaluate(model,loaders['val'],device,m['classes']);v=metrics(d['predictions'],m['classes'])
                history.append(dict(epoch=epoch,train_loss=loss_sum/total,**{k:v[k] for k in ['accuracy','recall_mean','f1_mean']}))
                save(folder/'history.json',history)
                if v['f1_mean']>best:
                    best=v['f1_mean'];save(folder/'best_val.json',v)
                    torch.save(dict(state_dict={k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
                        method=name,modality=role,classes=m['classes'],seed=args.seed,best_epoch=epoch,
                        manifest_sha256=hashlib.sha256(raw).hexdigest()),folder/'best.pt')
                scheduler.step()
                print(f'{name}/{role} epoch={epoch} VAL n=157 Acc={100*v["accuracy"]:.2f} Recall(mean)={100*v["recall_mean"]:.2f} F1(mean)={100*v["f1_mean"]:.2f}',flush=True)
            record(name,role,json.loads((folder/'best_val.json').read_text()))
            del model,optimizer,scheduler;torch.cuda.empty_cache()
    if len(table)!=9:raise ValueError('Expected eight single-sensor baselines plus dual GFM')
    if path.read_bytes()!=raw or any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Source files changed')
    save(args.output/'comparison.json',table)
    # Re-express existing frozen noise predictions; never re-evaluate or train.
    if args.noise_results:
        old=json.loads((args.noise_results/'results.json').read_text());converted=[]
        for r in old:
            d=json.loads((args.noise_results/f'{r["method"]}_seed{r["seed"]}'/f'val_{r["scenario"]}.json').read_text())
            v=metrics(d['predictions'],m['classes'])
            converted.append(dict(method=r['method'],scenario=r['scenario'],seed=r['seed'],n=v['n'],
                **{k:v[k] for k in ['accuracy','recall_mean','f1_mean']}))
        with (args.output/'existing_noise_ARF1.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(converted[0]));writer.writeheader();writer.writerows(converted)
    print('DONE: 9 clean validation rows (8 single-channel baselines + dual GFM), all n=157; no test images.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--noise-results',type=Path)
    p.add_argument('--seed',type=int,choices=[3407],default=3407);p.add_argument('--epochs',type=int,default=30)
    main(p.parse_args())
