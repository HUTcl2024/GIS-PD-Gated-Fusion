"""Train fair fusion ablations on one saved split; evaluate held-out test once.

All methods use identical preprocessing, splits, optimizer and selection rule.
No historical result is imported. Smoke outputs cannot be used in the paper.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from revision_data import ROOT, dump, image_paths, region_paths


METHODS=['uhf','hfct','concat','equal','gfm','gfm_no_conf',
         'recovered_concat','recovered_equal','swin_uhf','swin_hfct','gfm_attn','region_equal','region_gfm','cnn_gfm']
AUX_METHODS={'concat','equal','gfm','gfm_no_conf','gfm_attn','region_equal','region_gfm','cnn_gfm'}


def build_model(method,backbone,dim,classes,pretrained):
    if method in {'recovered_concat','recovered_equal','swin_uhf','swin_hfct'}:
        from recovered_models import build_recovered
        return build_recovered(method,backbone,dim,classes,pretrained)
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


class MultiRegionEncoder(nn.Module):
    """Four independent image encoders: global, 0-120, 120-240, 240-360."""
    def __init__(self,backbone,dim,classes,pretrained):
        super().__init__()
        self.branches=nn.ModuleList()
        for _ in range(4):
            weights=(models.ResNet50_Weights.IMAGENET1K_V1 if backbone=='resnet50' else models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else None
            base=getattr(models,backbone)(weights=weights); n=base.fc.in_features; base.fc=nn.Identity()
            self.branches.append(nn.Sequential(base,nn.Linear(n,dim),nn.ReLU()))
        self.merge=nn.Sequential(nn.Linear(dim*4,dim),nn.ReLU())
        self.head=nn.Sequential(nn.Dropout(.5),nn.Linear(dim,classes))

    def forward(self,x):
        if x.ndim!=5 or x.shape[1]!=4: raise ValueError('Expected B x 4 views x 3 x H x W')
        feature=self.merge(torch.cat([branch(x[:,i]) for i,branch in enumerate(self.branches)],dim=1))
        return feature,self.head(feature)


class GateFeatureAttention(nn.Module):
    """Per-feature gate input scaling, initially the identity; not a quality oracle."""
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,max(1,dim//16)),nn.ReLU(),nn.Linear(max(1,dim//16),dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,features):
        return features*(2*self.net(features).sigmoid())


class FusionNet(nn.Module):
    def __init__(self,method,backbone,dim,classes,pretrained=True):
        super().__init__()
        self.method=method
        # Both encoders initialized in the same order across all methods/seeds.
        encoder=CNNEncoder if method=='cnn_gfm' else MultiRegionEncoder if method in ['region_equal','region_gfm'] else Encoder
        self.uhf=encoder(backbone,dim,classes,pretrained)
        self.hfct=encoder(backbone,dim,classes,pretrained)
        self.gate=nn.Sequential(nn.Linear(dim*2+2,128),nn.ReLU(),nn.Linear(128,2))
        self.concat=nn.Linear(dim*2,classes)
        # Only the new method has these parameters; original checkpoints remain compatible.
        if method=='gfm_attn':
            self.gate_attn_uhf=GateFeatureAttention(dim)
            self.gate_attn_hfct=GateFeatureAttention(dim)

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
        elif self.method in ['equal','region_equal']:
            logits=(lu+lh)/2
        else:
            cu=lu.softmax(1).amax(1,keepdim=True); ch=lh.softmax(1).amax(1,keepdim=True)
            if self.method=='gfm_no_conf': cu=torch.zeros_like(cu); ch=torch.zeros_like(ch)
            if self.method=='gfm_attn':
                fu=self.gate_attn_uhf(fu); fh=self.gate_attn_hfct(fh)
            w=self.gate(torch.cat([fu,fh,cu,ch],1)).softmax(1)
            logits=w[:,:1]*lu+w[:,1:]*lh
        if return_branches: return logits,w,lu,lh
        return logits,w


def training_objective(logits,labels,aux_weight=0.,branch_logits=None):
    """L = CE(fused,y) + lambda * [CE(UHF,y)+CE(HFCT,y)]/2.

    A zero coefficient follows the old loss path exactly. It does not create
    zero gradients for otherwise-unused branch heads (important for concat).
    """
    fused_loss=nn.functional.cross_entropy(logits,labels)
    if aux_weight==0: return fused_loss,fused_loss,logits.new_zeros(())
    if branch_logits is None or len(branch_logits)!=2:
        raise ValueError('Auxiliary supervision requires both branch logits')
    auxiliary=sum(nn.functional.cross_entropy(z,labels) for z in branch_logits)/2
    return fused_loss+aux_weight*auxiliary,fused_loss,auxiliary


class Paired(Dataset):
    def __init__(self,manifest,root,split,scenario='clean',limit=None):
        self.records=[r for r in manifest['records'] if r['split']==split]
        if limit:
            selected=[]
            for c in manifest['classes']:
                candidates=[r for r in self.records if r['class_name']==c]
                if manifest.get('representation')=='legacy_phase_tiles_30deg':
                    parents=list(dict.fromkeys(r['parent_record_id'] for r in candidates))[:limit]
                    selected.extend(r for r in candidates if r['parent_record_id'] in parents)
                else: selected.extend(candidates[:limit])
            self.records=selected
        if manifest.get('representation')=='legacy_phase_tiles_30deg':
            groups={}
            for r in manifest['records']: groups.setdefault(r['group_id'],[]).append(r)
            for children in groups.values():
                if (len(children)!=12 or len({r['split'] for r in children})!=1 or
                        len({r['class_name'] for r in children})!=1 or
                        len({r['parent_record_id'] for r in children})!=1 or
                        {r['phase_start_deg'] for r in children}!=set(range(0,360,30)) or
                        any(r['phase_stop_deg']!=r['phase_start_deg']+30 for r in children)):
                    raise ValueError('Every tile group requires all 12 phase blocks in one split/class')
            if len({r['record_id'] for r in manifest['records']})!=len(manifest['records']):
                raise ValueError('Duplicate tile record IDs')
        self.labels={c:i for i,c in enumerate(manifest['classes'])}
        self.region_views=manifest.get('representation')=='legacy_global_three_regions'
        if self.region_views:
            if (len({r['group_id'] for r in manifest['records']})!=len(manifest['records'])
                    or len({r['record_id'] for r in manifest['records']})!=len(manifest['records'])):
                raise ValueError('Region records must be unique original parents')
            if [(v['start_deg'],v['stop_deg']) for v in manifest['views']]!=[(0,360),(0,120),(120,240),(240,360)]:
                raise ValueError('Unexpected region view ordering')
        path_builder=region_paths if self.region_views else image_paths
        self.paths=[path_builder(root,scenario,r) for r in self.records]
        missing=[str(p) for pair in self.paths for p in pair if not p.is_file()]
        if missing: raise FileNotFoundError(f'{len(missing)} missing PRPD images; first: {missing[0]}')
        # Phase patterns must not be flipped or randomly cropped.
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __len__(self): return len(self.records)

    def __getitem__(self,i):
        images=[]
        for p in self.paths[i]:
            with Image.open(p) as im: images.append(self.transform(im.convert('RGB')))
        if self.region_views: images=[torch.stack(images[:4]),torch.stack(images[4:])]
        return *images,self.labels[self.records[i]['class_name']],self.records[i]['record_id']


class OnlineNoisePaired(Paired):
    """Fresh raw-waveform perturbations per parent/epoch; PNGs held in RAM only."""
    def __init__(self,manifest,root,raw_root,probability):
        super().__init__(manifest,root,'train')
        self.raw_root=raw_root; self.references=manifest['preprocessing']['a_ref']
        self.probability=probability; self.cache={}; self.parents={}
        for r in self.records: self.parents.setdefault(r['parent_record_id'],dict(r,record_id=r['parent_record_id']))

    def set_epoch(self,epoch,seed):
        from revision_data import online_parent_images
        self.cache.clear(); chosen=[]
        for parent in sorted(self.parents):
            value=int.from_bytes(hashlib.sha256(f'online-choice/{seed}/{epoch}/{parent}'.encode()).digest()[:8],'little')
            rng=np.random.default_rng(value)
            if rng.random()<self.probability:
                chosen.append((parent,float(rng.uniform(-15,10)),str(rng.choice(['both','uhf','hfct']))))
        for i,(parent,snr,target) in enumerate(chosen,1):
            images=online_parent_images(self.raw_root,self.parents[parent],self.references,snr,target,
                noise_seed=f'train-online/{seed}/{epoch}')
            self.cache.update(images)
            if i==1 or i%10==0 or i==len(chosen):
                mb=sum(len(value) for pair in self.cache.values() for value in pair.values())/1024**2
                print(f'online noise seed={seed} epoch={epoch} [{i}/{len(chosen)}] RAM PNG={mb:.1f} MiB',flush=True)

    def __getitem__(self,i):
        from io import BytesIO
        r=self.records[i]; pair=self.cache.get(r['record_id'])
        if pair is None: return super().__getitem__(i)
        images=[]
        for role in ['uhf','hfct']:
            with Image.open(BytesIO(pair[role])) as image: images.append(self.transform(image.convert('RGB')))
        return *images,self.labels[r['class_name']],r['record_id']


class NoiseBankPaired(Paired):
    """Same paired parent-level corruption decisions for every compared method."""
    def __init__(self,manifest,root,probability):
        super().__init__(manifest,root,'train')
        config=json.loads((root/'images'/'train_awgn_bank_v1'/'config.json').read_text(encoding='utf-8'))
        if config.get('split')!='train' or config.get('tile_manifest_sha256')!=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Training noise bank split/manifest mismatch')
        expected={r['parent_record_id'] for r in self.records}
        if set(config['plan'])!=expected: raise ValueError('Training bank parent set mismatch')
        self.noisy_paths=[image_paths(root,'train_awgn_bank_v1',r) for r in self.records]
        for pair in self.noisy_paths:
            for path in pair:
                if not path.is_file(): raise FileNotFoundError(path)
        self.probability=probability; self.clean_paths=self.paths; self.set_epoch(1,3407)

    def set_epoch(self,epoch,seed):
        decisions={}
        for r in self.records:
            parent=r['parent_record_id']
            if parent not in decisions:
                value=int.from_bytes(hashlib.sha256(f'noise-choice/{seed}/{epoch}/{parent}'.encode()).digest()[:8],'little')/2**64
                decisions[parent]=value<self.probability
        self.paths=[self.noisy_paths[i] if decisions[r['parent_record_id']] else self.clean_paths[i]
            for i,r in enumerate(self.records)]


class CachedBranches(Dataset):
    """RAM-only frozen backbone features/logits; no derived cache files."""
    def __init__(self,u,h,y,ids):
        self.u,self.h,self.y,self.ids=u,h,y,ids

    def __len__(self): return len(self.y)

    def __getitem__(self,i): return self.u[i],self.h[i],self.y[i],self.ids[i]


class FrozenFeatureFusion(nn.Module):
    """Only projections and gate train; logits and backbone features are fixed.

    The single-channel CE never trained feat_proj in stage 1. Stage 2 therefore
    trains those projections, while neither classifier nor backbone can change.
    """
    def __init__(self,method,raw_dim,dim,classes):
        super().__init__()
        self.method,self.raw_dim=method,raw_dim
        self.uhf_proj=nn.Sequential(nn.Linear(raw_dim,dim),nn.ReLU())
        self.hfct_proj=nn.Sequential(nn.Linear(raw_dim,dim),nn.ReLU())
        self.gate=nn.Sequential(nn.Linear(dim*2+2,128),nn.ReLU(),nn.Linear(128,2))

    def forward(self,u,h,return_branches=False):
        lu,lh=u[:,self.raw_dim:],h[:,self.raw_dim:]
        w=None
        if self.method=='equal': logits=(lu+lh)/2
        else:
            fu=self.uhf_proj(u[:,:self.raw_dim]); fh=self.hfct_proj(h[:,:self.raw_dim])
            cu=lu.softmax(1).amax(1,keepdim=True); ch=lh.softmax(1).amax(1,keepdim=True)
            if self.method=='gfm_no_conf': cu=torch.zeros_like(cu); ch=torch.zeros_like(ch)
            w=self.gate(torch.cat([fu,fh,cu,ch],1)).softmax(1)
            logits=w[:,:1]*lu+w[:,1:]*lh
        return (logits,w,lu,lh) if return_branches else (logits,w)


@torch.no_grad()
def cache_branches(encoders,loader,device):
    for encoder in encoders: encoder.eval()
    us,hs,ys,ids=[],[],[],[]
    for u,h,y,record_ids in loader:
        packed=[]
        for encoder,x in zip(encoders,[u,h]):
            feature=encoder.backbone(x.to(device))
            packed.append(torch.cat([feature,encoder.head(feature)],1).cpu())
        us.append(packed[0]); hs.append(packed[1]); ys.append(y); ids.extend(record_ids)
    cached=CachedBranches(torch.cat(us),torch.cat(hs),torch.cat(ys),ids)
    if hasattr(loader.dataset,'records'): cached.records=loader.dataset.records
    return cached


def run_two_stage(args,manifest,raw,device):
    """Reuse independently trained CE branch checkpoints as completed stage 1.

    Read only their weights, protocol and selection metadata, never test scores.
    Compare all stage-2 methods on identical cached train/validation records.
    """
    source=args.branch_source
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    manifest_hash=hashlib.sha256(raw).hexdigest()
    if protocol['manifest_sha256']!=manifest_hash:
        raise ValueError('Stage-1 checkpoints used a different data manifest')
    if protocol['arguments'].get('smoke',False) or protocol['arguments'].get('aux_weight',0)!=0:
        raise ValueError('Stage 1 requires the independent full-data single-channel CE checkpoints')
    if protocol.get('preprocessing')!='Resize 224 RGB ImageNet normalization; no augmentation':
        raise ValueError('Stage-1 preprocessing differs from the current pipeline')
    datasets={s:Paired(manifest,args.data,s,limit=2 if args.smoke else None) for s in ['train','val']}
    # Fail before creating output if any expected branch checkpoint is absent.
    for seed in args.seeds:
        for role in ['uhf','hfct']:
            if not (source/f'{role}_seed{seed}'/'best.pt').is_file():
                raise FileNotFoundError(f'Missing stage-1 {role} seed {seed}')
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        manifest_sha256=manifest_hash,classes=manifest['classes'],evaluation_split='val',
        training_samples={s:len(d) for s,d in datasets.items()},
        preprocessing=protocol['preprocessing'],stage1_source=str(source),
        stage1='Reuse independently CE-trained branches selected by validation macro F1',
        stage2='Freeze backbone and classifier including BN/dropout in eval mode; train projections and gate with fused CE only',
        cache='RAM only; raw backbone features and fixed logits; no test images',
        loss='CE(fused,y)',selection='validation macro F1; first maximum wins',
        torch=torch.__version__,torchvision=__import__('torchvision').__version__,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warning='SMOKE ONLY' if args.smoke else 'DEVELOPMENT ONLY: selected and evaluated on validation; not independent test evidence'))
    results=[]
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        encoders=[]; origins=[]; backbone=None
        for role in ['uhf','hfct']:
            path=source/f'{role}_seed{seed}'/'best.pt'
            c=torch.load(path,map_location='cpu',weights_only=True)
            if c['method']!=role or c['seed']!=seed or c['classes']!=manifest['classes']:
                raise ValueError('Stage-1 identity/classes mismatch')
            if backbone is not None and (backbone,dim)!=(c['backbone'],c['feat_dim']):
                raise ValueError('Branch architectures differ')
            backbone,dim=c['backbone'],c['feat_dim']
            encoder=Encoder(backbone,dim,len(c['classes']),False)
            encoder.load_state_dict({k[len(role)+1:]:v for k,v in c['state_dict'].items() if k.startswith(role+'.')},strict=True)
            encoder.requires_grad_(False); encoder.eval(); encoders.append(encoder.to(device))
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
            origins.append(dict(role=role,path=str(path),sha256=digest.hexdigest(),best_epoch=c['best_epoch']))
            del c
        cached={s:cache_branches(encoders,DataLoader(d,batch_size=args.batch_size,shuffle=False),device) for s,d in datasets.items()}
        raw_dim=cached['train'].u.shape[1]-len(manifest['classes'])
        torch.manual_seed(seed)
        initial=FrozenFeatureFusion('gfm',raw_dim,dim,len(manifest['classes']))
        initial.uhf_proj.load_state_dict(encoders[0].feat_proj.state_dict())
        initial.hfct_proj.load_state_dict(encoders[1].feat_proj.state_dict())
        initial_state=copy.deepcopy(initial.state_dict()); del initial,encoders,encoder
        print(f'stage1 reused seed={seed}; frozen train/val cache ready',flush=True)
        for method in args.methods:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model=FrozenFeatureFusion(method,raw_dim,dim,len(manifest['classes'])).to(device)
            model.load_state_dict(initial_state,strict=True)
            generator=torch.Generator().manual_seed(seed)
            train_loader=DataLoader(cached['train'],batch_size=args.batch_size,shuffle=True,generator=generator)
            val_loader=DataLoader(cached['val'],batch_size=args.batch_size,shuffle=False)
            folder=args.output/f'{method}_seed{seed}'; folder.mkdir()
            history=[]; best=-1.; best_epoch=0; start=time.perf_counter()
            if method!='equal':
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
                for epoch in range(1,(1 if args.smoke else args.epochs)+1):
                    model.train(); loss_sum=0.; count=0
                    for u,h,y,_ in train_loader:
                        optimizer.zero_grad(set_to_none=True)
                        logits,_=model(u.to(device),h.to(device))
                        loss=nn.functional.cross_entropy(logits,y.to(device)); loss.backward(); optimizer.step()
                        loss_sum+=float(loss.detach())*len(y); count+=len(y)
                    val=evaluate(model,val_loader,device,manifest['classes'],diagnostics=True)
                    history.append(dict(epoch=epoch,train_loss=loss_sum/count,val_macro_f1=val['macro_f1'],
                                        val_accuracy=val['accuracy'],branch_accuracy=val['branch_accuracy'],gate=val.get('gate')))
                    if val['macro_f1']>best:
                        best=val['macro_f1']; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch
                    scheduler.step()
                    g=val['gate']
                    print(f'two-stage {method} seed={seed} epoch={epoch} val_f1={val["macro_f1"]:.4f} sat(U/H)={g["uhf_saturated_fraction"]:.3f}/{g["hfct_saturated_fraction"]:.3f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
                model.load_state_dict(best_state); del optimizer,scheduler,best_state
                # Only save small stage-2 weights; stage-1 weights stay shared.
                torch.save(dict(state_dict=model.state_dict(),method=method,backbone=backbone,feat_dim=dim,
                    raw_dim=raw_dim,classes=manifest['classes'],seed=seed,best_epoch=best_epoch,
                    two_stage=True,validation_only=True,branch_sources=origins),folder/'best.pt')
            metrics=evaluate(model,val_loader,device,manifest['classes'],diagnostics=True)
            metrics.update(evaluation_split='val',selection_used_this_set=True,branch_sources=origins)
            dump(folder/'history.json',history); dump(folder/'val_clean.json',metrics)
            results.append(dict(method=method,seed=seed,scenario='val_clean',evaluation_split='val',
                **{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']},
                branch_accuracy=metrics['branch_accuracy'],gate=metrics.get('gate')))
            dump(args.output/'results.json',results)
            print(f'two-stage {method} seed={seed} selected_epoch={best_epoch} validation_f1={metrics["macro_f1"]:.4f}',flush=True)
            del model
        del cached,initial_state
        if device.type=='cuda': torch.cuda.empty_cache()
    summarize(args.output,results)


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
    metadata=getattr(loader.dataset,'records',None)
    if getattr(loader.dataset,'region_views',False):
        result.update(metric_unit='original_record',pooling='one joint prediction from eight views; no tile probability averaging')
    if metadata and 'parent_record_id' in metadata[0]:
        parent=aggregate_tiles(rows,metadata,classes)
        parent['tile_level']={k:v for k,v in result.items() if k!='predictions'}
        parent['tile_predictions']=rows
        if 'gate' in result:
            parent['gate']=result['gate']; parent['gate_unit']='phase_tile'
        return parent
    return result


def aggregate_tiles(rows,metadata,classes):
    """Prespecified arithmetic mean of 12 class-probability vectors per parent.

    Record-level F1 is the selection metric; tile-level scores remain available.
    Reject partial/duplicate parents instead of silently averaging fewer tiles.
    """
    lookup={r['record_id']:r for r in metadata}
    if len({r['record_id'] for r in rows})!=len(rows) or set(lookup)!={r['record_id'] for r in rows}:
        raise ValueError('Prediction IDs do not match the complete tile dataset')
    groups={}
    for row in rows:
        item=lookup[row['record_id']]
        if classes[row['true']]!=item['class_name']: raise ValueError('Tile label differs from manifest class')
        groups.setdefault(item['group_id'],[]).append((row,item))
    parents=[]
    for group,items in groups.items():
        if (len(items)!=12 or len({r['true'] for r,_ in items})!=1 or
                {m['phase_start_deg'] for _,m in items}!=set(range(0,360,30))):
            raise ValueError('Record aggregation requires 12 unique phase blocks with identical labels')
        probability=np.mean([r['probabilities'] for r,_ in items],axis=0)
        row=dict(record_id=items[0][1]['parent_record_id'],group_id=group,true=items[0][0]['true'],
                 pred=int(probability.argmax()),probabilities=probability.tolist(),tile_count=12,
                 weights=np.mean([r['weights'] for r,_ in items],axis=0).tolist() if items[0][0]['weights'] is not None else None)
        if 'branch_probabilities' in items[0][0]:
            bp=np.mean([r['branch_probabilities'] for r,_ in items],axis=0)
            row.update(branch_probabilities=bp.tolist(),branch_predictions=bp.argmax(1).tolist(),
                       branch_confidences=bp.max(1).tolist())
        parents.append(row)
    y=[r['true'] for r in parents]; pred=[r['pred'] for r in parents]; labels=list(range(len(classes)))
    result=dict(n=len(y),correct=sum(a==b for a,b in zip(y,pred)),metric_unit='original_record',
        pooling='arithmetic mean of the 12 tile softmax probability vectors',
        accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=labels,average='macro',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,pred),
        confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist(),
        report=classification_report(y,pred,labels=labels,target_names=classes,zero_division=0,output_dict=True),predictions=parents)
    if 'branch_predictions' in parents[0]:
        result['branch_accuracy']={role:sum(r['branch_predictions'][i]==r['true'] for r in parents)/len(parents)
                                   for i,role in enumerate(['uhf','hfct'])}
    return result


def evaluate_saved_validation(args,manifest,raw,device):
    """Reuse existing checkpoints without training or touching their saved results."""
    if not args.evaluate_only or args.eval_split!='val' or args.validation_only or args.two_stage or args.aux_weight:
        raise ValueError('checkpoint-source requires --evaluate-only --eval-split val, without training flags')
    source=args.checkpoint_source.resolve()
    if args.output.resolve()==source: raise ValueError('Evaluation output must differ from checkpoint source')
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest(): raise ValueError('Source manifest mismatch')
    for scenario in args.scenarios:
        if scenario!='clean' and not scenario.startswith('val_'): raise ValueError('Validation noise scenario requires val_ prefix')
        if scenario!='clean':
            config=json.loads((args.data/'images'/scenario/'config.json').read_text(encoding='utf-8'))
            if config.get('split')!='val': raise ValueError('Noise scenario split mismatch')
    for method in args.methods:
        for seed in args.seeds:
            if not (source/f'{method}_seed{seed}'/'best.pt').is_file(): raise FileNotFoundError(f'{method} seed {seed}')
    settings=dict(checkpoint_source=str(source),source_protocol=protocol,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),evaluation_split='val',methods=args.methods,seeds=args.seeds,
        selection='unchanged source checkpoints; no retraining or noise-based model selection',
        metric=('one original-record prediction from eight views' if manifest.get('representation')=='legacy_global_three_regions' else
            'original-record mean of 12 tile probabilities; tile metrics also retained'),
        warning='Validation robustness analysis; not an independent final test result')
    args.output.mkdir(parents=True,exist_ok=True)
    config_path=args.output/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=settings:
        raise ValueError('Evaluation output protocol mismatch')
    dump(config_path,settings)
    results_path=args.output/'results.json'
    results=json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
    for method in args.methods:
        for seed in args.seeds:
            ckpt=torch.load(source/f'{method}_seed{seed}'/'best.pt',map_location=device,weights_only=True)
            if ckpt['method']!=method or ckpt['seed']!=seed or ckpt['classes']!=manifest['classes']:
                raise ValueError('Checkpoint identity mismatch')
            model=build_model(method,ckpt['backbone'],ckpt['feat_dim'],len(ckpt['classes']),False).to(device)
            model.load_state_dict(ckpt['state_dict'])
            for scenario in args.scenarios:
                destination=args.output/f'{method}_seed{seed}'/f'val_{scenario}.json'
                if destination.exists():
                    metrics=json.loads(destination.read_text(encoding='utf-8'))
                else:
                    ds=Paired(manifest,args.data,'val',scenario,2 if protocol['arguments'].get('smoke') else None)
                    metrics=evaluate(model,DataLoader(ds,batch_size=args.batch_size),device,manifest['classes'],
                        diagnostics=method in AUX_METHODS)
                    dump(destination,metrics)
                key=(method,seed,scenario)
                results=[r for r in results if (r['method'],r['seed'],r['scenario'])!=key]
                results.append(dict(method=method,seed=seed,scenario=scenario,evaluation_split='val',
                    metric_unit=metrics.get('metric_unit','image'),
                    **{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']},
                    **{f'tile_{k}':metrics['tile_level'][k] for k in ['n','accuracy','macro_f1']} if 'tile_level' in metrics else {}))
                dump(results_path,results)
                print(f'{method} seed={seed} {scenario} records={metrics["n"]} acc={metrics["accuracy"]:.4f} f1={metrics["macro_f1"]:.4f}',flush=True)
            del model,ckpt
    summarize(args.output,results)


@torch.inference_mode()
def feature_tsne(args,manifest,raw,device):
    """Record-level feature diagnostics; no model fitting or test-set access."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if (args.checkpoint_source is None or len(args.seeds)!=1 or args.eval_split!='val'
            or args.evaluate_only or args.validation_only or args.two_stage or args.aux_weight):
        raise ValueError('feature-tsne requires checkpoint-source, one seed, eval-split val, no training flags')
    source=args.checkpoint_source
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest(): raise ValueError('Manifest mismatch')
    seed=args.seeds[0]; checkpoint=source/f'gfm_seed{seed}'/'best.pt'
    ckpt=torch.load(checkpoint,map_location=device,weights_only=True)
    if ckpt['method']!='gfm' or ckpt['seed']!=seed or ckpt['classes']!=manifest['classes']:
        raise ValueError('Checkpoint identity mismatch')
    if args.output.resolve()==source.resolve(): raise ValueError('Use a separate figure output directory')
    if any((args.output/f'tsne_seed{seed}.{ext}').exists() for ext in ['png','npz']):
        raise FileExistsError('Feature figure already exists; preserve previous output')
    model=build_model('gfm',ckpt['backbone'],ckpt['feat_dim'],len(ckpt['classes']),False).to(device)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    scenarios=['clean','val_awgn_uhf_snr-15dB','val_awgn_hfct_snr-15dB','val_awgn_both_snr-15dB']
    roles=['uhf','hfct']; features=[]; parent_ids=None; labels=None
    for scenario in scenarios:
        if scenario!='clean':
            config=json.loads((args.data/'images'/scenario/'config.json').read_text(encoding='utf-8'))
            if config.get('split')!='val': raise ValueError('Expected validation noise')
        ds=Paired(manifest,args.data,'val',scenario)
        metadata={r['record_id']:r for r in ds.records}; groups={}
        for u,h,y,ids in DataLoader(ds,batch_size=args.batch_size,shuffle=False):
            fu=model.uhf.feat_proj(model.uhf.backbone(u.to(device))).cpu().numpy()
            fh=model.hfct.feat_proj(model.hfct.backbone(h.to(device))).cpu().numpy()
            for i,rid in enumerate(ids):
                r=metadata[rid]; p=r['parent_record_id']
                groups.setdefault(p,[]).append((fu[i],fh[i],int(y[i])))
        order=sorted(groups); current_labels=np.array([groups[p][0][2] for p in order])
        if any(len(v)!=12 or len({r[2] for r in v})!=1 for v in groups.values()):
            raise ValueError('Incomplete or inconsistent feature group')
        if parent_ids is None: parent_ids=order; labels=current_labels
        elif order!=parent_ids or not np.array_equal(labels,current_labels): raise ValueError('Feature parent mismatch')
        features.append(np.array([[np.mean([r[k] for r in groups[p]],axis=0) for p in order] for k in [0,1]]))
        print(f'Extracted {scenario}: {len(order)} parents, projected features {ckpt["feat_dim"]}D, mean of 12 tiles.',flush=True)
    features=np.stack(features,axis=1)  # branch, condition, parent, dimension
    embeddings=[]; scores=[]; n=len(parent_ids)
    if n<=30: raise ValueError('At least 31 parents required for fixed perplexity 30')
    print('branch | condition | silhouette in original feature space | mean relative feature drift',flush=True)
    for branch,role in enumerate(roles):
        f=features[branch]
        pca=PCA(n_components=min(50,n-1,f.shape[-1]),svd_solver='full').fit(f[0])
        reduced=pca.transform(f.reshape(-1,f.shape[-1]))
        embedding=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=seed).fit_transform(reduced)
        embeddings.append(embedding.reshape(len(scenarios),n,2))
        branch_scores=[]
        for index,scenario in enumerate(scenarios):
            score=float(silhouette_score(f[index],labels)); branch_scores.append(score)
            drift=float(np.mean(np.linalg.norm(f[index]-f[0],axis=1)/(np.linalg.norm(f[0],axis=1)+1e-12)))
            print(f'{role} | {scenario} | {score:.4f} | {drift:.4f}',flush=True)
        scores.append(branch_scores)
    embeddings=np.array(embeddings)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(12,6),squeeze=False)
    titles=['Clean','UHF noise (-15 dB)','HFCT noise (-15 dB)','Both noisy (-15 dB)']
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    for b,role in enumerate(roles):
        xy=embeddings[b]; lo=xy.min(axis=(0,1)); hi=xy.max(axis=(0,1)); pad=(hi-lo)*.06
        for index in range(4):
            ax=axes[b,index]
            for label,name in enumerate(manifest['classes']):
                mask=labels==label
                ax.scatter(xy[index,mask,0],xy[index,mask,1],s=14,c=colors[label],label=name,alpha=.8,edgecolors='none')
            ax.set(title=f'{titles[index]}\n{role.upper()} | silhouette={scores[b][index]:.3f}',
                xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),xticks=[],yticks=[])
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=len(legend),frameon=False)
    fig.suptitle(f'GFM seed {seed} | {n} validation records | projected {ckpt["feat_dim"]}D features\nJoint t-SNE within each branch; coordinates are not comparable between branches',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.91))
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/f'tsne_seed{seed}.png',dpi=220)
    plt.close(fig)
    info=dict(seed=seed,selected_epoch=ckpt['best_epoch'],layer='feat_proj after ReLU',pooling='mean of 12 tile features',
        evaluation_split='val',pca='50 components maximum; fit on clean validation features separately per branch',
        tsne='joint clean and noisy within branch; perplexity 30; init PCA; learning rate auto',
        manifest_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        warning='Exploratory visualization, not proof of collapse or generalisation')
    np.savez_compressed(args.output/f'tsne_seed{seed}.npz',features=features,embedding=embeddings,labels=labels,
        parent_ids=np.array(parent_ids),classes=np.array(manifest['classes']),scenarios=np.array(scenarios),metadata=json.dumps(info))
    print(f'Saved one PNG and its feature data in {args.output}. Inspect the image before drawing conclusions.',flush=True)


def confidence_entropy_weights(lu,lh,alpha=.5,entropy_weight=1.,center_logits=False):
    """Route A: entropy of softmax probabilities in nats, no temporal input."""
    if not 0<=alpha<=1 or not np.isfinite(entropy_weight) or entropy_weight<0:
        raise ValueError('alpha in [0,1] and nonnegative finite entropy weight required')
    scores=[]
    for logits in [lu,lh]:
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite branch logits')
        logp=logits.log_softmax(1); p=logp.exp()
        entropy=-(p*logp).sum(1)
        z=logits-logits.mean(1,keepdim=True) if center_logits else logits
        scores.append(alpha*z.amax(1)+(1-alpha)*p.amax(1)-entropy_weight*entropy)
    return torch.stack(scores,1).softmax(1)


class PhaseRuleProbe(nn.Module):
    """Change only fusion of the two logits from this same 30-degree tile."""
    def __init__(self,model,mode,alpha,entropy_weight):
        super().__init__(); self.model=model; self.mode=mode
        self.alpha=alpha; self.entropy_weight=entropy_weight
    def forward(self,u,h,return_branches=False):
        original,w,lu,lh=self.model(u,h,return_branches=True)
        if self.mode=='original_gfm': logits=original
        else:
            w=(torch.full((len(lu),2),.5,device=lu.device,dtype=lu.dtype) if self.mode=='fixed_equal' else
                confidence_entropy_weights(lu,lh,self.alpha,self.entropy_weight,self.mode=='rule_centered'))
            logits=w[:,:1]*lu+w[:,1:]*lh
        return (logits,w,lu,lh) if return_branches else (logits,w)


@torch.inference_mode()
def evaluate_phase_rule(args,manifest,raw,device):
    if (manifest.get('representation')!='legacy_phase_tiles_30deg' or args.eval_split!='val'
            or args.checkpoint_source is None or args.evaluate_only or args.validation_only or args.two_stage
            or args.aux_weight or args.online_noise or args.train_noise_bank or args.feature_tsne or args.gfm_counterfactual):
        raise ValueError('phase-rule requires 30-degree tile data, checkpoint-source, eval-split val and no training flags')
    if not np.isfinite(args.rule_alpha) or not 0<=args.rule_alpha<=1 or not np.isfinite(args.rule_lambda) or args.rule_lambda<0:
        raise ValueError('Finite alpha in [0,1] and lambda>=0 required')
    source=args.checkpoint_source
    if source.resolve()==args.output.resolve(): raise ValueError('Use a separate output directory')
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest(): raise ValueError('Manifest differs from source training')
    if protocol['arguments'].get('online_noise') or protocol['arguments'].get('train_noise_bank'):
        raise ValueError('Route A baseline source must be clean-trained')
    datasets={}
    for scenario in args.scenarios:
        if scenario!='clean':
            if not scenario.startswith('val_'): raise ValueError('Only fixed validation scenarios supported')
            config=json.loads((args.data/'images'/scenario/'config.json').read_text(encoding='utf-8'))
            if config.get('split')!='val': raise ValueError('Wrong scenario split')
        datasets[scenario]=Paired(manifest,args.data,'val',scenario)
    for seed in args.seeds:
        if not (source/f'gfm_seed{seed}'/'best.pt').is_file(): raise FileNotFoundError(f'GFM source seed {seed}')
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        evaluation_split='val',manifest_sha256=hashlib.sha256(raw).hexdigest(),source_protocol=protocol,
        alpha=args.rule_alpha,entropy_weight=args.rule_lambda,
        formula='score=alpha*max(logits)+(1-alpha)*max(softmax(logits))-lambda*H(softmax(logits)); weights=softmax(scores)',
        entropy='natural logarithm, nats; not entropy of raw logits',
        centered_control='subtract each branch logit mean ONLY in the max-logit score term',
        input='same 30-degree paired tile, no adjacent tiles, time frames or buffers',
        metric='mean of 12 tile class probabilities per original record; tile metrics retained',
        warning='Fixed-parameter validation diagnostic; same GFM internal branches for all rules, not independently trained equal baseline'))
    modes=['original_gfm','fixed_equal','rule_raw','rule_centered']; results=[]
    for seed in args.seeds:
        path=source/f'gfm_seed{seed}'/'best.pt'
        ckpt=torch.load(path,map_location=device,weights_only=True)
        if ckpt['method']!='gfm' or ckpt['seed']!=seed or ckpt['classes']!=manifest['classes']: raise ValueError('Checkpoint identity mismatch')
        model=build_model('gfm',ckpt['backbone'],ckpt['feat_dim'],len(ckpt['classes']),False).to(device)
        model.load_state_dict(ckpt['state_dict']); model.requires_grad_(False); model.eval()
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        origin=dict(path=str(path),sha256=digest.hexdigest(),selected_epoch=ckpt['best_epoch'])
        print(f'Route A frozen checkpoint: {path}; alpha={args.rule_alpha}, lambda={args.rule_lambda}',flush=True)
        for scenario,ds in datasets.items():
            for mode in modes:
                probe=PhaseRuleProbe(model,mode,args.rule_alpha,args.rule_lambda).eval()
                metrics=evaluate(probe,DataLoader(ds,batch_size=args.batch_size),device,manifest['classes'],diagnostics=True)
                metrics.update(evaluation_split='val',frozen_source=origin)
                dump(args.output/f'{mode}_seed{seed}'/f'val_{scenario}.json',metrics)
                results.append(dict(method=mode,seed=seed,scenario=scenario,evaluation_split='val',
                    **{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']}))
                dump(args.output/'results.json',results)
                print(f'{mode} seed={seed} {scenario} records={metrics["n"]} acc={metrics["accuracy"]:.4f} '
                    f'tile_acc={metrics["tile_level"]["accuracy"]:.4f}',flush=True)
        del model,ckpt
    print('Done. No training, raw waveform processing or source-checkpoint changes.',flush=True)


class FixedFusionProbe(nn.Module):
    """Read-only inference intervention on one already-trained GFM."""
    def __init__(self,model,mode):
        super().__init__(); self.model=model; self.mode=mode

    def forward(self,u,h):
        logits,w,lu,lh=self.model(u,h,return_branches=True)
        if self.mode=='learned': return logits,w
        if self.mode=='equal': return (lu+lh)/2,torch.full_like(w,.5)
        if self.mode=='hfct': return lh,torch.stack([torch.zeros_like(w[:,0]),torch.ones_like(w[:,0])],1)
        if self.mode=='uhf': return lu,torch.stack([torch.ones_like(w[:,0]),torch.zeros_like(w[:,0])],1)
        raise ValueError('Unknown inference intervention')


@torch.inference_mode()
def probe_gfm(args,manifest,raw,device):
    if args.evaluate_only or args.validation_only or args.two_stage or args.aux_weight or args.eval_split!='val':
        raise ValueError('gfm-counterfactual requires --eval-split val without other training/evaluation mode flags')
    source=args.checkpoint_source or args.output
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest(): raise ValueError('Source manifest mismatch')
    for scenario in args.scenarios:
        if scenario=='clean': continue
        if not scenario.startswith('val_'): raise ValueError('Only validation scenarios are allowed')
        config=json.loads((args.data/'images'/scenario/'config.json').read_text(encoding='utf-8'))
        if config.get('split')!='val': raise ValueError('Noise scenario split mismatch')
    print('READ ONLY: same trained GFM; no optimization, checkpoint changes or result files.',flush=True)
    print('Internal single-channel branches belong to this GFM; they are not separately trained baseline models.',flush=True)
    for seed in args.seeds:
        path=source/f'gfm_seed{seed}'/'best.pt'
        ckpt=torch.load(path,map_location=device,weights_only=True)
        if ckpt['method']!='gfm' or ckpt['seed']!=seed or ckpt['classes']!=manifest['classes']:
            raise ValueError('Checkpoint identity mismatch')
        model=build_model('gfm',ckpt['backbone'],ckpt['feat_dim'],len(ckpt['classes']),False).to(device)
        model.load_state_dict(ckpt['state_dict']); model.eval()
        print(f'Checkpoint: {path}; selected epoch: {ckpt["best_epoch"]}',flush=True)
        print('scenario | records | learned acc% (correct) | forced equal acc% (correct) | internal HFCT acc% | internal UHF acc% | equal-learned pp',flush=True)
        for scenario in args.scenarios:
            ds=Paired(manifest,args.data,'val',scenario)
            loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
            scores={mode:evaluate(FixedFusionProbe(model,mode).eval(),loader,device,manifest['classes'])
                for mode in ['learned','equal','hfct','uhf']}
            a,b,h,u=[scores[mode] for mode in ['learned','equal','hfct','uhf']]
            print(f'{scenario} | {a["n"]} | {100*a["accuracy"]:.2f} ({a["correct"]}) | '
                f'{100*b["accuracy"]:.2f} ({b["correct"]}) | {100*h["accuracy"]:.2f} | '
                f'{100*u["accuracy"]:.2f} | {100*(b["accuracy"]-a["accuracy"]):+.2f}',flush=True)
        del model,ckpt
    print('Done. Positive equal-learned means forcing equal weights improved this fixed GFM.',flush=True)


def main(args):
    if args.online_noise and (not args.validation_only or args.raw_data is None or args.train_noise_bank or args.evaluate_only
            or args.two_stage or args.gfm_counterfactual or args.feature_tsne or args.checkpoint_source or args.smoke):
        raise ValueError('online-noise requires raw-data and ordinary full validation-only training, without other modes')
    if args.train_noise_bank and (not args.validation_only or args.evaluate_only or args.two_stage or args.gfm_counterfactual or args.feature_tsne or args.checkpoint_source):
        raise ValueError('Training noise augmentation requires ordinary --validation-only training')
    if not np.isfinite(args.noise_probability) or not 0<=args.noise_probability<=1:
        raise ValueError('noise-probability must be in [0,1]')
    if args.selection_unit=='tile' and args.two_stage:
        raise ValueError('Tile selection is supported by ordinary training only')
    if getattr(args,'two_stage',False):
        if (not args.validation_only or args.evaluate_only or args.aux_weight!=0 or args.branch_source is None
                or not set(args.methods)<={'equal','gfm','gfm_no_conf'} or args.scenarios!=['clean']):
            raise ValueError('Two-stage requires --validation-only, --branch-source, aux-weight 0, clean only, and methods equal/gfm/gfm_no_conf')
    if not np.isfinite(args.aux_weight) or args.aux_weight<0:
        raise ValueError('aux-weight must be finite and nonnegative')
    if args.aux_weight>0 and (not args.validation_only or args.evaluate_only):
        raise ValueError('New auxiliary-loss experiments must use --validation-only, without --evaluate-only')
    if args.aux_weight>0 and not set(args.methods)<=AUX_METHODS:
        raise ValueError('Use only concat, equal, gfm, gfm_no_conf for matched auxiliary supervision')
    if args.validation_only and (args.evaluate_only or args.scenarios!=['clean']):
        raise ValueError('Validation-only development cannot evaluate test noise scenarios')
    raw=(args.data/'manifest.json').read_bytes(); manifest=json.loads(raw)
    region_methods={'region_equal','region_gfm'}
    region_input=manifest.get('representation')=='legacy_global_three_regions'
    if region_input:
        if not set(args.methods)<=region_methods or args.online_noise or args.train_noise_bank or args.two_stage or args.feature_tsne or args.gfm_counterfactual:
            raise ValueError('Region data currently supports region_equal/region_gfm only, without other experiment modes')
    elif set(args.methods)&region_methods: raise ValueError('Region models require a region-fit manifest and region-render images')
    if args.selection_unit=='tile' and manifest.get('representation')!='legacy_phase_tiles_30deg':
        raise ValueError('Tile selection requires the legacy phase-tile manifest')
    splits={s:{r['group_id'] for r in manifest['records'] if r['split']==s} for s in ['train','val','test']}
    if any(splits[a]&splits[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Independent group leakage')
    for s in splits:
        if {r['class_name'] for r in manifest['records'] if r['split']==s}!=set(manifest['classes']):
            raise ValueError('Missing class in '+s)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:',device,flush=True)
    torch.set_num_threads(4)
    if args.phase_rule:
        evaluate_phase_rule(args,manifest,raw,device)
        return
    if args.feature_tsne:
        feature_tsne(args,manifest,raw,device)
        return
    if args.gfm_counterfactual:
        probe_gfm(args,manifest,raw,device)
        return
    if args.checkpoint_source is not None:
        evaluate_saved_validation(args,manifest,raw,device)
        return
    if args.eval_split!='test': raise ValueError('eval-split val requires checkpoint-source')
    if getattr(args,'two_stage',False):
        run_two_stage(args,manifest,raw,device)
        return
    if args.evaluate_only:
        protocol=json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
        if protocol['arguments'].get('validation_only',False):
            raise ValueError('This is a validation-only development run; do not evaluate it on the already-inspected test set')
        if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest():
            raise ValueError('Manifest changed since training')
        results_path=args.output/'results.json'
        results=json.loads(results_path.read_text(encoding='utf-8'))
        for folder in sorted(args.output.glob('*_seed*')):
            ckpt=torch.load(folder/'best.pt',map_location=device,weights_only=True)
            model=build_model(ckpt['method'],ckpt['backbone'],ckpt['feat_dim'],len(ckpt['classes']),False).to(device)
            model.load_state_dict(ckpt['state_dict'])
            for scenario in args.scenarios:
                destination=folder/f'test_{scenario}.json'
                if destination.exists(): continue
                ds=Paired(manifest,args.data,'test',scenario,2 if protocol['arguments']['smoke'] else None)
                metrics=evaluate(model,DataLoader(ds,batch_size=args.batch_size),device,manifest['classes'])
                dump(destination,metrics)
                results.append(dict(method=ckpt['method'],seed=ckpt['seed'],scenario=scenario,
                    **{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']}))
                dump(results_path,results)
            del model
        summarize(args.output,results)
        return
    active_splits=['train','val'] if args.validation_only else list(splits)
    datasets={s:Paired(manifest,args.data,s,limit=2 if args.smoke else None) for s in active_splits}
    if args.train_noise_bank:
        if args.smoke: raise ValueError('Use the complete noise bank, not a partial smoke run')
        datasets['train']=NoiseBankPaired(manifest,args.data,args.noise_probability)
    if args.online_noise: datasets['train']=OnlineNoisePaired(manifest,args.data,args.raw_data,args.noise_probability)
    backbone='resnet18' if args.smoke else 'resnet50'; dim=128 if args.smoke else 512
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        manifest_sha256=hashlib.sha256(raw).hexdigest(),classes=manifest['classes'],backbone=backbone,feat_dim=dim,
        model_backbones={method:'simple_cnn' if method=='cnn_gfm' else 'swin_t' if method.startswith('swin_') else backbone for method in args.methods},
        initialization={method:('random CNN; no pretrained weights' if method=='cnn_gfm' else 'random' if args.no_pretrained else 'ImageNet') for method in args.methods},
        gate_feature_attention={method:('independent modality MLPs dim->dim/16->dim, ReLU, 2*sigmoid feature scaling; identity init; gate inputs only; branch logits unchanged'
            if method=='gfm_attn' else 'none') for method in args.methods},
        source_code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                            for name in ['revision_train.py','recovered_models.py'] if (ROOT/name).exists()},
        torch=torch.__version__,torchvision=__import__('torchvision').__version__,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        preprocessing='Resize 224 RGB ImageNet normalization; no augmentation',
        loss='CE(fused,y) + aux_weight * (CE(UHF,y)+CE(HFCT,y))/2' if args.aux_weight>0 else 'fused cross entropy; equal fusion averages logits and is trained jointly',
        evaluation_split='val' if args.validation_only else 'test',
        representation=manifest.get('representation','whole_record_prpd'),
        multi_region=(dict(views=manifest['views'],encoders='4 independent ResNets per modality; 8 total',
            per_view_dim=dim,within_modality=f'concat {dim*4}->Linear {dim}->ReLU; Dropout 0.5->Linear classes',
            between_modalities='equal logits or original confidence+feature gate; no CBAM') if region_input else None),
        noise_augmentation=(dict(probability=args.noise_probability,unit='parent record; all 12 tiles paired',
            selection='clean validation only; same criterion as clean-trained baseline',
            bank=json.loads((args.data/'images'/'train_awgn_bank_v1'/'config.json').read_text(encoding='utf-8')))
            if args.train_noise_bank else (dict(probability=args.noise_probability,unit='original training parent, all 12 paired tiles',
                snr='uniform continuous [-15,10] dB',target='uniform both/uhf/hfct',
                resampling='fresh decision, SNR, target and AWGN every epoch; identical per seed/epoch across methods',
                storage='one epoch compressed PNG bytes in RAM only',phase='CH2 unperturbed',
                normalization='unchanged training references',selection='same clean validation criterion as baseline') if args.online_noise else None)),
        selection=('tile validation macro F1; first maximum wins; record metrics also reported' if args.selection_unit=='tile' else
            'original-record validation macro F1; first maximum wins; 12 tile probabilities averaged for tile manifests'),
        training_samples={s:len(d) for s,d in datasets.items()},
        warning='SMOKE ONLY' if args.smoke else ('DEVELOPMENT: validation set used for selection; not a test result' if args.validation_only else 'Experimental results, not yet reviewed')))
    results=[]
    for seed in args.seeds:
        for method in args.methods:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
            generator=torch.Generator().manual_seed(seed)
            loaders={s:DataLoader(d,batch_size=args.batch_size,shuffle=(s=='train'),num_workers=0,
                                 generator=generator if s=='train' else None) for s,d in datasets.items()}
            model=build_model(method,backbone,dim,len(manifest['classes']),not args.no_pretrained).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
            folder=args.output/f'{method}_seed{seed}'; folder.mkdir()
            best=-1; history=[]; start=time.perf_counter()
            for epoch in range(1,(1 if args.smoke else args.epochs)+1):
                if args.train_noise_bank or args.online_noise: datasets['train'].set_epoch(epoch,seed)
                model.train(); total_loss=0; total_fused=0; total_aux=0; total=0
                for u,h,y,_ in loaders['train']:
                    optimizer.zero_grad(set_to_none=True)
                    if args.aux_weight>0:
                        logits,_,lu,lh=model(u.to(device),h.to(device),return_branches=True)
                        branches=(lu,lh)
                    else:
                        logits,_=model(u.to(device),h.to(device)); branches=None
                    loss,fused_loss,auxiliary=training_objective(logits,y.to(device),args.aux_weight,branches)
                    loss.backward(); optimizer.step()
                    total_loss+=float(loss.detach())*len(y); total+=len(y)
                    total_fused+=float(fused_loss.detach())*len(y); total_aux+=float(auxiliary.detach())*len(y)
                diagnostics=args.validation_only and method in AUX_METHODS
                val=evaluate(model,loaders['val'],device,manifest['classes'],diagnostics=diagnostics)
                history.append(dict(epoch=epoch,train_loss=total_loss/total,val_accuracy=val['accuracy'],val_macro_f1=val['macro_f1']))
                if 'tile_level' in val:
                    history[-1].update(val_tile_macro_f1=val['tile_level']['macro_f1'],val_records=val['n'],val_tiles=val['tile_level']['n'])
                if args.validation_only:
                    history[-1].update(train_fused_loss=total_fused/total,train_auxiliary_loss=total_aux/total,
                                       branch_accuracy=val.get('branch_accuracy'),gate=val.get('gate'))
                selection_f1=val['tile_level']['macro_f1'] if args.selection_unit=='tile' else val['macro_f1']
                if selection_f1>best:
                    best=selection_f1; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch
                scheduler.step()
                detail=''
                detail+=f' selection={args.selection_unit}'
                if 'tile_level' in val: detail+=f' tile_f1={val["tile_level"]["macro_f1"]:.4f} records={val["n"]}'
                if diagnostics:
                    b=val['branch_accuracy']; g=val.get('gate')
                    detail+=f' branch_acc(U/H)={b["uhf"]:.3f}/{b["hfct"]:.3f}'
                    if g: detail+=f' sat(U/H)={g["uhf_saturated_fraction"]:.3f}/{g["hfct_saturated_fraction"]:.3f}'
                print(f'{method} seed={seed} epoch={epoch} val_f1={val["macro_f1"]:.4f}{detail} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            model.load_state_dict(best_state)
            torch.save(dict(state_dict=best_state,method=method,backbone='simple_cnn' if method=='cnn_gfm' else 'swin_t' if method.startswith('swin_') else backbone,feat_dim=dim,
                classes=manifest['classes'],seed=seed,best_epoch=best_epoch,
                aux_weight=args.aux_weight,validation_only=args.validation_only,selection_unit=args.selection_unit),folder/'best.pt')
            dump(folder/'history.json',history)
            if args.validation_only:
                metrics=evaluate(model,loaders['val'],device,manifest['classes'],diagnostics=method in AUX_METHODS)
                metrics.update(evaluation_split='val',selection_used_this_set=True)
                dump(folder/'val_clean.json',metrics)
                row=dict(method=method,seed=seed,scenario='val_clean',evaluation_split='val',
                    **{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']})
                row.update(branch_accuracy=metrics.get('branch_accuracy'),gate=metrics.get('gate'))
                if 'tile_level' in metrics:
                    row.update(tile_n=metrics['tile_level']['n'],tile_accuracy=metrics['tile_level']['accuracy'],
                               tile_macro_f1=metrics['tile_level']['macro_f1'],metric_unit='original_record')
                results.append(row); dump(args.output/'results.json',results)
                del model,best_state,optimizer
                if device.type=='cuda': torch.cuda.empty_cache()
                continue
            for scenario in args.scenarios:
                loader=loaders['test'] if scenario=='clean' else DataLoader(Paired(manifest,args.data,'test',scenario,2 if args.smoke else None),batch_size=args.batch_size)
                metrics=evaluate(model,loader,device,manifest['classes'])
                dump(folder/f'test_{scenario}.json',metrics)
                results.append(dict(method=method,seed=seed,scenario=scenario,**{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']}))
                dump(args.output/'results.json',results)
            del model,best_state,optimizer
            if device.type=='cuda': torch.cuda.empty_cache()
    summarize(args.output,results)


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
            if 'tile_macro_f1' in rows[0]:
                summary[-1].update(metric_unit='original_record',tile_macro_f1_mean=float(np.mean([r['tile_macro_f1'] for r in rows])),
                                   tile_accuracy_mean=float(np.mean([r['tile_accuracy'] for r in rows])))
    dump(output/'summary.json',summary)
    paired=[]
    for scenario in sorted({r['scenario'] for r in results}):
        by_key={(r['method'],r['seed']):r for r in results if r['scenario']==scenario}
        for comparison in ['concat','equal','gfm_no_conf']:
            deltas=[dict(seed=seed,accuracy_delta_pp=100*(by_key['gfm',seed]['accuracy']-by_key[comparison,seed]['accuracy']))
                    for seed in sorted({r['seed'] for r in results}) if ('gfm',seed) in by_key and (comparison,seed) in by_key]
            if deltas: paired.append(dict(scenario=scenario,comparison='gfm minus '+comparison,runs=deltas))
    dump(output/'paired_differences.json',paired)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'revision_output')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS[:6])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--selection-unit',choices=['record','tile'],default='record',help='Validation unit for checkpoint selection; tile reproduces earlier tile-selected runs')
    p.add_argument('--scenarios',nargs='+',default=['clean'])
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-pretrained',action='store_true',help='Explicit offline random initialization; record this in paper')
    p.add_argument('--evaluate-only',action='store_true',help='Evaluate saved checkpoints on additional fixed test scenarios; no training')
    p.add_argument('--gfm-counterfactual',action='store_true',help='Read-only validation probe: original gate, forced equal, internal HFCT/UHF branches of the same GFM')
    p.add_argument('--phase-rule',action='store_true',help='Frozen 30-degree-tile confidence/entropy gating rule, no training or temporal input')
    p.add_argument('--rule-alpha',type=float,default=.5)
    p.add_argument('--rule-lambda',type=float,default=1.)
    p.add_argument('--feature-tsne',action='store_true',help='Plot clean and -15 dB validation projected features from one fixed GFM')
    p.add_argument('--train-noise-bank',action='store_true',help='Train using the fixed raw-waveform AWGN bank with paired parent-level random selection each epoch')
    p.add_argument('--online-noise',action='store_true',help='Resample raw-waveform noise and render paired training tiles each epoch, held in RAM only')
    p.add_argument('--raw-data',type=Path,help='Original WFM/CSV root for online-noise training')
    p.add_argument('--noise-probability',type=float,default=.5,help='Probability of choosing the noisy variant per training parent per epoch')
    p.add_argument('--checkpoint-source',type=Path,help='Reuse saved models for validation robustness; write to a separate output directory')
    p.add_argument('--eval-split',choices=['val','test'],default='test')
    p.add_argument('--aux-weight',type=float,default=0.,help='Weight of mean branch CE; new objective requires --validation-only')
    p.add_argument('--validation-only',action='store_true',help='Develop on train/val only; do not load or evaluate test images')
    p.add_argument('--two-stage',action='store_true',help='Reuse independent branches, freeze them, and develop fusion on train/val')
    p.add_argument('--branch-source',type=Path,help='Existing single-channel checkpoint directory, e.g. runs/formal')
    main(p.parse_args())
