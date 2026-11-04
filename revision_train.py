"""Clean full360 ResNet50 baselines and confidence-gated fusion.

Original state-dict names and initialization order are retained. Training uses
train/val only; fixed clean/noise test evaluation is in scripts/. No tile,
region, temporal, attention, reliability or noise-training modes remain.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from PIL import Image
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
from revision_data import ROOT,dump,image_paths
METHODS=['uhf','hfct','concat','equal','gfm','gfm_no_conf']
AUX_METHODS={'concat','equal','gfm','gfm_no_conf'}

def build_model(method,backbone,dim,classes,pretrained):
    if method in ['swin_uhf','swin_hfct']:
        from recovered_models import build_recovered
        return build_recovered(method,backbone,dim,classes,pretrained)
    if method not in METHODS: raise ValueError('Unsupported current method: '+method)
    return FusionNet(method,backbone,dim,classes,pretrained)

class Encoder(nn.Module):
    def __init__(self, backbone, dim, classes, pretrained):
        super().__init__()
        weights = (models.ResNet50_Weights.IMAGENET1K_V1 if backbone=='resnet50'
                   else models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else None
        self.backbone=getattr(models,backbone)(weights=weights)
        n=self.backbone.fc.in_features
        self.backbone.fc=nn.Identity()
        self.feat_proj=nn.Sequential(nn.Linear(n,dim),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(n,dim),nn.ReLU(),nn.Dropout(.5),nn.Linear(dim,classes))

    def forward(self,x):
        f=self.backbone(x)
        return self.feat_proj(f),self.head(f)

class CNNEncoder(nn.Module):
    """Three-block RGB CNN; retains the original GFM's feature/head interfaces."""
    def __init__(self, backbone, dim, classes, pretrained):
        super().__init__()
        layers = []
        previous = 3
        for width in [32, 64, 128]:
            layers += [nn.Conv2d(previous, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
            previous = width
        self.backbone = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feat_proj = nn.Sequential(nn.Linear(128, dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, dim), nn.ReLU(), nn.Dropout(.5), nn.Linear(dim, classes))

    def forward(self, x):
        feature = self.backbone(x)
        return self.feat_proj(feature), self.head(feature)

class FusionNet(nn.Module):
    def __init__(self,method,backbone,dim,classes,pretrained=True):
        super().__init__()
        self.method=method
        # Both encoders initialized in the same order across all methods/seeds.
        encoder=Encoder
        self.uhf=encoder(backbone,dim,classes,pretrained)
        self.hfct=encoder(backbone,dim,classes,pretrained)
        self.gate=nn.Sequential(nn.Linear(dim*2+2,128),nn.ReLU(),nn.Linear(128,2))
        self.concat=nn.Linear(dim*2,classes)

    def forward(self,u,h,return_branches=False):
        if return_branches and self.method not in AUX_METHODS:
            raise ValueError('Branch outputs are supported only for matched fusion methods')
        if self.method=='uhf':
            _,logits=self.uhf(u); return logits,None
        if self.method=='hfct':
            _,logits=self.hfct(h); return logits,None
        fu,lu=self.uhf(u); fh,lh=self.hfct(h)
        w=None
        if self.method=='concat':
            logits=self.concat(torch.cat([fu,fh],1))
        elif self.method=='equal':
            logits=(lu+lh)/2
        else:
            cu=lu.softmax(1).amax(1,keepdim=True); ch=lh.softmax(1).amax(1,keepdim=True)
            if self.method=='gfm_no_conf': cu=torch.zeros_like(cu); ch=torch.zeros_like(ch)
            w=self.gate(torch.cat([fu,fh,cu,ch],1)).softmax(1)
            logits=w[:,:1]*lu+w[:,1:]*lh
        if return_branches: return logits,w,lu,lh
        return logits,w

class Paired(Dataset):
    def __init__(self,manifest,root,split,scenario='clean',limit=None):
        if manifest.get('representation')!='whole_record_prpd' or any('parent_record_id' in r for r in manifest['records']):
            raise ValueError('Only complete360 original-record inputs are supported')
        self.records=[r for r in manifest['records'] if r['split']==split]
        if limit:
            self.records=[r for c in manifest['classes'] for r in [r for r in self.records if r['class_name']==c][:limit]]
        self.labels={c:i for i,c in enumerate(manifest['classes'])}
        self.paths=[image_paths(root,scenario,r) for r in self.records]
        missing=[str(p) for pair in self.paths for p in pair if not p.is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} missing PRPD images; first: {missing[0]}')
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        images=[]
        for p in self.paths[i]:
            with Image.open(p) as im:images.append(self.transform(im.convert('RGB')))
        return *images,self.labels[self.records[i]['class_name']],self.records[i]['record_id']


@torch.no_grad()
def evaluate(model,loader,device,classes,diagnostics=False):
    model.eval(); truth=[]; pred=[]; rows=[]
    branch_correct=[0,0]; gate_weights=[]
    for u,h,y,ids in loader:
        if diagnostics:
            logits,weights,lu,lh=model(u.to(device),h.to(device),return_branches=True)
            branch_pred=[z.argmax(1).cpu() for z in [lu,lh]]
            branch_conf=[z.softmax(1).amax(1).cpu().tolist() for z in [lu,lh]]
            branch_prob=[z.softmax(1).cpu().tolist() for z in [lu,lh]]
            for k in range(2): branch_correct[k]+=int((branch_pred[k]==y).sum())
            if weights is not None: gate_weights.extend(weights.cpu().tolist())
        else:
            logits,weights=model(u.to(device),h.to(device))
        p=logits.argmax(1).cpu().tolist()
        truth.extend(y.tolist()); pred.extend(p)
        probabilities=logits.softmax(1).cpu().tolist()
        for j,rid in enumerate(ids):
            rows.append(dict(record_id=rid,true=int(y[j]),pred=p[j],probabilities=probabilities[j],
                weights=weights[j].cpu().tolist() if weights is not None else None))
            if diagnostics:
                rows[-1].update(branch_predictions=[int(z[j]) for z in branch_pred],
                                branch_confidences=[z[j] for z in branch_conf],
                                branch_probabilities=[z[j] for z in branch_prob])
    result=dict(n=len(truth),correct=sum(a==b for a,b in zip(truth,pred)),
        accuracy=accuracy_score(truth,pred),macro_f1=f1_score(truth,pred,labels=list(range(len(classes))),average='macro',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(truth,pred),
        confusion_matrix=confusion_matrix(truth,pred,labels=list(range(len(classes)))).tolist(),
        report=classification_report(truth,pred,labels=list(range(len(classes))),target_names=classes,zero_division=0,output_dict=True),
        predictions=rows)
    if diagnostics:
        result['branch_accuracy']={role:branch_correct[i]/len(truth) for i,role in enumerate(['uhf','hfct'])}
        if gate_weights:
            w=np.asarray(gate_weights)
            result['gate']=dict(mean_uhf_weight=float(w[:,0].mean()),
                uhf_saturated_fraction=float((w[:,0]>=.99).mean()),
                hfct_saturated_fraction=float((w[:,1]>=.99).mean()))
    result['metric_unit']='original_record'
    return result

def summarize(output,results):
    summary=[]
    for method in sorted({r['method'] for r in results}):
        for scenario in sorted({r['scenario'] for r in results}):
            rows=[r for r in results if r['method']==method and r['scenario']==scenario]
            if not rows: continue
            summary.append(dict(method=method,scenario=scenario,runs=len(rows),
                evaluation_split=rows[0].get('evaluation_split','test'),
                selection_used_this_set=rows[0].get('evaluation_split','test')=='val',
                accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),
                accuracy_sd=float(np.std([r['accuracy'] for r in rows],ddof=1)) if len(rows)>1 else None,
                macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows])),
                macro_f1_sd=float(np.std([r['macro_f1'] for r in rows],ddof=1)) if len(rows)>1 else None))
    dump(output/'summary.json',summary)
    paired=[]
    for scenario in sorted({r['scenario'] for r in results}):
        by_key={(r['method'],r['seed']):r for r in results if r['scenario']==scenario}
        for comparison in ['concat','equal','gfm_no_conf']:
            deltas=[dict(seed=seed,accuracy_delta_pp=100*(by_key['gfm',seed]['accuracy']-by_key[comparison,seed]['accuracy']))
                    for seed in sorted({r['seed'] for r in results}) if ('gfm',seed) in by_key and (comparison,seed) in by_key]
            if deltas: paired.append(dict(scenario=scenario,comparison='gfm minus '+comparison,runs=deltas))
    dump(output/'paired_differences.json',paired)

def main(args):
    raw=(args.data/'manifest.json').read_bytes();m=json.loads(raw)
    if args.output.exists():raise FileExistsError('Use a fresh output directory')
    if len(m['records'])!=786 or len({r['group_id'] for r in m['records']})!=786:raise ValueError('Expected786 unique original records')
    if dict(Counter(r['split'] for r in m['records']))!={'train':471,'val':157,'test':158}:raise ValueError('Unexpected split')
    if m['preprocessing'].get('amplitude_normalization') is not False:raise ValueError('Expected raw-voltage PRPD')
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);print('device:',device,flush=True)
    datasets={s:Paired(m,args.data,s) for s in ['train','val']}
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        manifest_sha256=hashlib.sha256(raw).hexdigest(),classes=m['classes'],backbone='resnet50',feat_dim=512,
        representation='whole_record_prpd',selection='Clean record-level validation macro F1; first maximum',
        preprocessing='RGB224; ImageNet pixel normalization; no augmentation',
        optimizer='Adam lr0.001; StepLR step10 gamma0.1',loss='Fused cross entropy only',
        source_code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['revision_train.py','recovered_models.py']},
        evaluation_split='val',training_samples={s:len(d) for s,d in datasets.items()},
        torch=torch.__version__,torchvision=__import__('torchvision').__version__))
    results=[]
    for seed in args.seeds:
        for method in args.methods:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            generator=torch.Generator().manual_seed(seed)
            loaders={s:DataLoader(d,batch_size=args.batch_size,shuffle=s=='train',num_workers=0,
                generator=generator if s=='train' else None) for s,d in datasets.items()}
            model=build_model(method,'resnet50',512,len(m['classes']),not args.no_pretrained).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
            folder=args.output/f'{method}_seed{seed}';folder.mkdir();history=[];best=-1.;start=time.perf_counter()
            for epoch in range(1,args.epochs+1):
                model.train();total_loss=0.;total=0
                for u,h,y,_ in loaders['train']:
                    optimizer.zero_grad(set_to_none=True)
                    logits,_=model(u.to(device),h.to(device));loss=nn.functional.cross_entropy(logits,y.to(device))
                    loss.backward();optimizer.step();total_loss+=float(loss.detach())*len(y);total+=len(y)
                val=evaluate(model,loaders['val'],device,m['classes'],diagnostics=method in AUX_METHODS)
                history.append(dict(epoch=epoch,train_loss=total_loss/total,val_accuracy=val['accuracy'],val_macro_f1=val['macro_f1'],
                    branch_accuracy=val.get('branch_accuracy'),gate=val.get('gate')))
                if val['macro_f1']>best:
                    best=val['macro_f1'];best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
                scheduler.step()
                print(f'{method} seed={seed} epoch={epoch} val_f1={val["macro_f1"]:.4f} records=157 elapsed={time.perf_counter()-start:.1f}s',flush=True)
            model.load_state_dict(best_state)
            torch.save(dict(state_dict=best_state,method=method,backbone='resnet50',feat_dim=512,classes=m['classes'],
                seed=seed,best_epoch=best_epoch,aux_weight=0.,validation_only=True,selection_unit='record'),folder/'best.pt')
            dump(folder/'history.json',history)
            val=evaluate(model,loaders['val'],device,m['classes'],diagnostics=method in AUX_METHODS)
            val.update(evaluation_split='val',selection_used_this_set=True);dump(folder/'val_clean.json',val)
            results.append(dict(method=method,seed=seed,scenario='val_clean',evaluation_split='val',
                **{k:val[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']}))
            dump(args.output/'results.json',results)
            del model,best_state,optimizer
            if device.type=='cuda':torch.cuda.empty_cache()
    summarize(args.output,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--seeds',nargs='+',type=int,default=[3407])
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--selection-unit',choices=['record'],default='record')
    p.add_argument('--aux-weight',type=float,choices=[0.],default=0.)
    p.add_argument('--validation-only',action='store_true',default=True)
    p.add_argument('--no-pretrained',action='store_true');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:p.error('Positive epochs/batch size required')
    main(args)
