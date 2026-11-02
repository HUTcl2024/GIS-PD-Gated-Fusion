"""Train gates only on frozen clean-trained multi-region experts.

All baselines share exactly the same frozen experts. Only train parents generate
new waveform perturbations; validation scenarios are fixed, and used for model
selection. These are development results, not independent final-test evidence.
"""
import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from revision_data import dump, online_parent_images, REGION_VIEWS
from revision_train import Paired, build_model, evaluate

SCENARIOS=['clean']+[f'val_awgn_{target}_snr{snr}dB'
    for target in ['both','uhf','hfct'] for snr in [-15,-10,-5,0,5,10]]
GATE_MODES=['gate_ce','gate_quality','gate_ce_logits','gate_quality_logits','gate_interval_logits','feature_concat']


class FrozenFeatures(Dataset):
    region_views=True
    def __init__(self,u,h,y,ids,records):
        self.u,self.h,self.y,self.ids,self.records=u,h,y,ids,records
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.u[i],self.h[i],self.y[i],self.ids[i]


class ReliabilityFusion(nn.Module):
    def __init__(self,dim,classes,mode):
        super().__init__(); self.dim,self.classes,self.mode=dim,classes,mode
        if mode=='feature_concat':
            self.classifier=nn.Linear(2*dim,classes)
            return
        self.gate=nn.Sequential(nn.Linear(2*dim+2*classes+6,64),nn.ReLU(),nn.Linear(64,2))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)

    def raw_gate_scores(self,u,h):
        fu,lu=u[:,:self.dim],u[:,self.dim:]
        fh,lh=h[:,:self.dim],h[:,self.dim:]
        pu,ph=lu.softmax(1),lh.softmax(1)
        statistics=[]
        for p in [pu,ph]:
            top=p.topk(2,dim=1).values
            statistics.extend([top[:,:1],top[:,:1]-top[:,1:2],
                -(p*p.clamp_min(1e-8).log()).sum(1,keepdim=True)/math.log(self.classes)])
        return self.gate(torch.cat([F.normalize(fu,dim=1),F.normalize(fh,dim=1),pu,ph]+statistics,dim=1))

    def forward(self,u,h,return_branches=False):
        fu,lu=u[:,:self.dim],u[:,self.dim:]
        fh,lh=h[:,:self.dim],h[:,self.dim:]
        if self.mode=='feature_concat':
            logits=self.classifier(torch.cat([fu,fh],dim=1))
            return (logits,None,lu,lh) if return_branches else (logits,None)
        pu,ph=lu.softmax(1),lh.softmax(1)
        if self.mode in ['equal_logits','equal_probs']:
            w=torch.full((len(u),2),.5,device=u.device,dtype=u.dtype)
        else:
            w=self.raw_gate_scores(u,h).softmax(1)
        logits=(w[:,:1]*lu+w[:,1:]*lh if self.mode=='equal_logits' or self.mode.endswith('_logits') else
            (w[:,:1]*pu+w[:,1:]*ph).clamp_min(1e-8).log())
        return (logits,w,lu,lh) if return_branches else (logits,w)


def gate_losses(logits,w,lu,lh,y):
    """Soft reliability target uses train labels only, never noise direction/SNR."""
    with torch.no_grad():
        log_true=torch.stack([lu.log_softmax(1).gather(1,y[:,None]).squeeze(1),
                              lh.log_softmax(1).gather(1,y[:,None]).squeeze(1)],dim=1)
        target=log_true.softmax(1)
    return F.cross_entropy(logits,y),F.kl_div(w.clamp_min(1e-8).log(),target,reduction='batchmean')


@torch.no_grad()
def training_weight_intervals(lu,lh,y):
    """Label-derived, positive-width winning intervals for TRAIN batches only.

    For each competing class k, require
      w*(lu[y]-lu[k]) + (1-w)*(lh[y]-lh[k]) >= 0.
    Targets are detached float64 geometry, not gate predictions. Zero-width
    tie-only solutions are excluded: a softmax gate cannot robustly target them.
    """
    if lu.ndim!=2 or lu.shape!=lh.shape or lu.shape[1]<2 or y.shape!=(len(lu),):
        raise ValueError('Invalid interval training batch')
    if not torch.isfinite(lu).all() or not torch.isfinite(lh).all():
        raise ValueError('Non-finite expert logits')
    if ((y<0)|(y>=lu.shape[1])).any(): raise ValueError('Invalid training label')
    a,b=lu.detach().double(),lh.detach().double()
    da=a.gather(1,y[:,None])-a; db=b.gather(1,y[:,None])-b
    slope=da-db
    boundary=-db/torch.where(slope!=0,slope,torch.ones_like(slope))
    lower=torch.where(slope>0,boundary,torch.full_like(boundary,-torch.inf)).amax(1).clamp_min(0.)
    upper=torch.where(slope<0,boundary,torch.full_like(boundary,torch.inf)).amin(1).clamp_max(1.)
    impossible=((slope==0)&(db<0)).any(1)
    valid=(lower<upper)&~impossible
    # Neutral bounds on infeasible records prevent infinities/NaNs downstream.
    lower=torch.where(valid,lower,torch.zeros_like(lower))
    upper=torch.where(valid,upper,torch.ones_like(upper))
    return lower,upper,valid


def interval_gate_loss(w,lu,lh,y):
    """Mean L1 distance to valid intervals; infeasible records contribute zero.

    This replaces reliability KL, coefficient 1. Classification CE remains on
    ALL records. No validation/test labels are used to create training targets.
    """
    lower,upper,valid=training_weight_intervals(lu,lh,y)
    weight=w[:,0].double()
    distance=F.relu(lower-weight)+F.relu(weight-upper)
    penalty=torch.where(valid,distance,torch.zeros_like(distance)).mean().to(w.dtype)
    stats=dict(interval_valid=int(valid.sum()),interval_total=len(y),
        interval_outside=int((valid&(distance>0)).sum()),
        interval_distance_sum=float(torch.where(valid,distance,torch.zeros_like(distance)).detach().sum()))
    return penalty,stats


@torch.no_grad()
def encode(model,u,h,device):
    model.eval()
    fu,lu=model.uhf(u.to(device)); fh,lh=model.hfct(h.to(device))
    return torch.cat([fu,lu],1).cpu(),torch.cat([fh,lh],1).cpu()


@torch.no_grad()
def cache_existing(model,dataset,device,batch_size):
    us=[]; hs=[]; ys=[]; ids=[]
    for u,h,y,rid in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        a,b=encode(model,u,h,device); us.append(a); hs.append(b); ys.append(y); ids.extend(rid)
    return FrozenFeatures(torch.cat(us),torch.cat(hs),torch.cat(ys),ids,dataset.records)


def noise_jobs(records,manifest,raw_root,seed,epoch):
    """Plan original RNG draws in original order, independently of workers."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,8419]))
    count=max(3,int(round(len(records)*.5)))
    indices=rng.choice(len(records),size=count,replace=False)
    targets=np.array(['both','uhf','hfct']*((count+2)//3))[:count]; rng.shuffle(targets)
    jobs=[]
    for index,target in zip(indices,targets):
        record=records[int(index)]
        if record['split']!='train': raise ValueError('Only training parents may generate perturbations')
        jobs.append((raw_root,record,manifest['preprocessing']['a_ref'],float(rng.uniform(-15,10)),
            str(target),f'reliability/{seed}/{epoch}'))
    return jobs


def render_noise_job(job):
    """Spawned CPU workers never receive models or initialize CUDA."""
    raw_root,record,references,snr,target,noise_seed=job
    return online_parent_images(raw_root,record,references,snr,target,
        noise_seed=noise_seed,view_definitions=REGION_VIEWS)


def noise_worker_init():
    torch.set_num_threads(1)


def ordered_noise_images(jobs,workers):
    """Bounded RAM prefetch; preserve exact sample order and per-record seeds."""
    if not workers:
        for job in jobs: yield job,render_noise_job(job)
        return
    # spawn avoids inheriting CUDA state from the parent. At most 2*workers
    # tasks may be running/queued; no PNG files or whole-epoch image cache.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=noise_worker_init) as pool:
        remaining=iter(jobs); pending=deque()
        for _ in range(2*workers):
            job=next(remaining,None)
            if job is None: break
            pending.append((job,pool.submit(render_noise_job,job)))
        while pending:
            job,future=pending.popleft()
            images=future.result()
            next_job=next(remaining,None)
            if next_job is not None: pending.append((next_job,pool.submit(render_noise_job,next_job)))
            yield job,images


def benchmark_noise(args):
    """Read train waveforms only; compare serial/parallel PNG identity in RAM."""
    if args.raw_data is None or args.noise_workers<1 or args.noise_benchmark<1:
        raise ValueError('Benchmark requires --raw-data, --noise-workers >=1, --noise-benchmark >=1')
    manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('representation')!='legacy_global_three_regions': raise ValueError('Expected region manifest')
    records=[r for r in manifest['records'] if r['split']=='train']
    jobs=noise_jobs(records,manifest,args.raw_data,args.seeds[0],1)[:args.noise_benchmark]
    if not jobs: raise ValueError('No training noise jobs')
    def digest(images):
        return {(rid,role):hashlib.sha256(content).hexdigest()
            for rid,pair in images.items() for role,content in pair.items()}
    print(f'CPU-only benchmark: {len(jobs)} train records; serial then {args.noise_workers} workers. No model loading, CUDA, training, val/test access or output files.',flush=True)
    started=time.perf_counter()
    reference=[digest(images) for _,images in ordered_noise_images(jobs,0)]
    serial=time.perf_counter()-started
    print(f'Serial seconds={serial:.2f}',flush=True)
    started=time.perf_counter()
    for index,(_,images) in enumerate(ordered_noise_images(jobs,args.noise_workers)):
        if digest(images)!=reference[index]: raise ValueError('Parallel PNG bytes differ; do not switch training')
    parallel=time.perf_counter()-started
    print(f'Parallel seconds={parallel:.2f}; same PNG bytes for every view/channel; observed render speed ratio={serial/parallel:.2f}x',flush=True)
    print('Timing includes worker startup and is affected by filesystem cache/other running jobs; not a whole-training speedup guarantee.',flush=True)


def epoch_noise(model,clean,manifest,raw_root,transform,device,seed,epoch,workers=0):
    """One fresh noisy variant for a random half of train parents, RAM only."""
    jobs=noise_jobs(clean.records,manifest,raw_root,seed,epoch); count=len(jobs)
    result={target:[] for target in ['both','uhf','hfct']}
    labels={c:i for i,c in enumerate(manifest['classes'])}
    for j,(job,images) in enumerate(ordered_noise_images(jobs,workers),1):
        _,record,_,_,target,_=job
        pair=[]
        for role in ['uhf','hfct']:
            views=[]
            for view,_,_ in REGION_VIEWS:
                with Image.open(BytesIO(images[record['record_id']+'_'+view][role])) as im:
                    views.append(transform(im.convert('RGB')))
            pair.append(torch.stack(views).unsqueeze(0))
        u,h=encode(model,*pair,device)
        result[str(target)].append((u[0],h[0],labels[record['class_name']]))
        if j==1 or j%20==0 or j==count:
            print(f'Gate train noise seed={seed} epoch={epoch} [{j}/{count}]',flush=True)
    return result


def balanced_batches(clean,noise,batch_size,seed,epoch):
    """50% clean; noisy directions differ by at most one per minibatch.

    Features are sampled with replacement; this is an explicit balanced gate
    sampler, not one pass through every parent. Both gate methods share batches.
    """
    if batch_size<8 or batch_size%2: raise ValueError('Use an even gate batch size >=8')
    if any(not noise[t] for t in ['both','uhf','hfct']): raise ValueError('Missing train-noise direction')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,2971]))
    half=batch_size//2
    for step in range(math.ceil(len(clean)/batch_size)):
        chosen=[(clean.u[i],clean.h[i],int(clean.y[i])) for i in rng.integers(0,len(clean),half)]
        counts=[half//3]*3
        for i in range(half%3): counts[(step+i)%3]+=1
        for target,n in zip(['both','uhf','hfct'],counts):
            chosen.extend(noise[target][int(i)] for i in rng.integers(0,len(noise[target]),n))
        rng.shuffle(chosen)
        yield torch.stack([v[0] for v in chosen]),torch.stack([v[1] for v in chosen]),torch.tensor([v[2] for v in chosen])


def audit_saved(args):
    """Read existing predictions only; do not load weights or raw waveforms."""
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    raw=(args.data/'manifest.json').read_bytes(); manifest=json.loads(raw)
    protocol=read(args.output/'protocol.json')
    if protocol.get('evaluation_split')!='val' or protocol.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Validation result protocol/manifest mismatch')
    expected={r['record_id']:manifest['classes'].index(r['class_name']) for r in manifest['records'] if r['split']=='val'}
    if not expected: raise ValueError('No validation parents')
    seeds=protocol['arguments']['seeds']
    methods=protocol['arguments'].get('gate_modes',['gate_ce','gate_quality'])
    print('READ ONLY: both channels corrupted; branch partitions and routing on the same original records.',flush=True)
    print('method | seed | condition | N | both wrong | only UHF correct | only HFCT correct | both correct | correct weight/one | wrong weight/one | tie/one | fused correct/one | fused correct/both-wrong',flush=True)
    for seed in seeds:
        for scenario in ['clean']+[f'val_awgn_both_snr{snr}dB' for snr in [-15,-10,-5,0,5,10]]:
            reference=None
            for method in methods:
                d=read(args.output/f'{method}_seed{seed}'/f'val_{scenario}.json')
                rows=d['predictions']; ids=[p['record_id'] for p in rows]
                if len(ids)!=len(set(ids)) or {p['record_id']:p['true'] for p in rows}!=expected or d['n']!=len(expected):
                    raise ValueError('Saved record set/labels mismatch')
                rows=sorted(rows,key=lambda p:p['record_id'])
                y=np.array([p['true'] for p in rows]); fused=np.array([p['pred'] for p in rows])
                probs=np.asarray([p['branch_probabilities'] for p in rows],dtype=float)
                w=np.asarray([p['weights'] for p in rows],dtype=float)
                if (probs.shape!=(len(y),2,len(manifest['classes'])) or w.shape!=(len(y),2)
                        or not np.isfinite(probs).all() or not np.isfinite(w).all()
                        or (probs<0).any() or (w<0).any()
                        or not np.allclose(probs.sum(2),1,atol=1e-5) or not np.allclose(w.sum(1),1,atol=1e-5)):
                    raise ValueError('Invalid probabilities or weights')
                if reference is None: reference=probs
                elif not np.allclose(reference,probs,atol=1e-7,rtol=1e-6):
                    raise ValueError('Frozen branch outputs differ between gate methods')
                correct=probs.argmax(2)==y[:,None]
                both_wrong=~correct.any(1); both_right=correct.all(1)
                only_u=correct[:,0]&~correct[:,1]; only_h=correct[:,1]&~correct[:,0]
                one=only_u|only_h; n_one=int(one.sum()); n_wrong=int(both_wrong.sum())
                preferred=w[one].argmax(1); good=correct[one].argmax(1)
                tie=np.abs(w[one,0]-w[one,1])<=1e-6
                hit=(preferred==good)&~tie; miss=(preferred!=good)&~tie
                def fraction(count,total):
                    return f'{count}/{total} ({100*count/total:.1f}%)' if total else 'NA (0 eligible)'
                values=[method,str(seed),scenario,str(len(y)),str(n_wrong),str(int(only_u.sum())),
                    str(int(only_h.sum())),str(int(both_right.sum())),
                    fraction(int(hit.sum()),n_one),fraction(int(miss.sum()),n_one),fraction(int(tie.sum()),n_one),
                    fraction(int(((fused==y)&one).sum()),n_one),fraction(int(((fused==y)&both_wrong).sum()),n_wrong)]
                print(' | '.join(values),flush=True)
    print('Correct weight means the only correct branch receives higher weight; it does not guarantee the fused class is correct.',flush=True)
    print('Both branch argmax predictions can be wrong while their probability mixture is correct; both-wrong is not a strict fusion-error bound.',flush=True)
    print('Frozen outputs matched across methods. No training, model loading, raw waveform access or file writes.',flush=True)


def attainable_weight(a,b,label):
    """Exact 1D linear-score feasibility via class crossings, not a coarse grid.

    Scores are w*a+(1-w)*b. Evaluate endpoints, crossings against the true class,
    and each intervening open interval. Argmax uses the first class on ties.
    """
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.ndim!=1 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid score vectors')
    if not 0<=label<len(a): raise ValueError('Invalid class')
    roots=[0.,1.]
    for c in range(len(a)):
        if c==label: continue
        intercept=b[label]-b[c]
        slope=(a[label]-a[c])-intercept
        if slope!=0:
            w=-intercept/slope
            if 0<=w<=1: roots.append(float(w))
    roots=sorted(set(roots))
    candidates=roots+[(lo+hi)/2 for lo,hi in zip(roots[:-1],roots[1:])]+[.5]
    feasible=False; strict=False
    tolerance=1e-12*max(1.,float(np.max(np.abs(np.r_[a,b]))))
    for w in candidates:
        scores=w*a+(1-w)*b
        if scores.argmax()==label:
            feasible=True
            margin=scores[label]-np.max(np.delete(scores,label))
            strict=strict or margin>tolerance
    return feasible,strict


def winning_interval(a,b,label):
    """Closed weight interval satisfying true score >= every other score.

    Width/distance describe score geometry only. Endpoints can be argmax ties;
    attainable_weight remains authoritative for first-class tie feasibility.
    """
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(np.r_[a,b]).all():
        raise ValueError('Invalid score vectors')
    if not 0<=label<len(a): raise ValueError('Invalid class')
    lo,hi=0.,1.
    for c in range(len(a)):
        if c==label: continue
        intercept=b[label]-b[c]; slope=(a[label]-a[c])-intercept
        if slope==0:
            if intercept<0: return None
        elif slope>0: lo=max(lo,float(-intercept/slope))
        else: hi=min(hi,float(-intercept/slope))
        if lo>hi: return None
    return lo,hi


def diagnosis_arrays(lu,lh,y,rows_by_method):
    """Read-only mechanistic diagnostics; no parameter fitting or label routing."""
    z=np.stack([lu,lh],axis=1).astype(np.float64)
    centered=z-z.mean(2,keepdims=True)
    scale=np.sqrt(np.mean(centered**2,axis=2))
    normalized=centered/np.maximum(scale[:,:,None],1e-8)
    shifted=z-z.max(2,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(2,keepdims=True)); p=np.exp(logp)
    confidence=p.max(2); entropy=-(p*logp).sum(2)
    correct=z.argmax(2)==y[:,None]
    pct=lambda mask:100*float(np.mean(mask)) if len(mask) else float('nan')
    mean=lambda values:float(np.mean(values)) if len(values) else float('nan')
    print('  BRANCH: role | Acc% | wrong N | Pmax correct/wrong | H(nats) correct/wrong | wrong with Pmax>=0.9 N | median centered RMS | raw max correct/wrong',flush=True)
    for k,role in enumerate(['uhf','hfct']):
        good=correct[:,k]; bad=~good
        print(f'    {role} | {pct(good):.2f} | {int(bad.sum())} | '
            f'{mean(confidence[good,k]):.4f}/{mean(confidence[bad,k]):.4f} | '
            f'{mean(entropy[good,k]):.4f}/{mean(entropy[bad,k]):.4f} | '
            f'{int((bad&(confidence[:,k]>=.9)).sum())} | {np.median(scale[:,k]):.4f} | '
            f'{mean(z[good,k].max(1)) if good.any() else float("nan"):.4f}/'
            f'{mean(z[bad,k].max(1)) if bad.any() else float("nan"):.4f}',flush=True)
    one=correct.sum(1)==1
    n_one=int(one.sum()); good_branch=correct[one].argmax(1)
    conf_tie=np.abs(confidence[one,0]-confidence[one,1])<=1e-8
    entropy_tie=np.abs(entropy[one,0]-entropy[one,1])<=1e-8
    conf_hit=(confidence[one].argmax(1)==good_branch)&~conf_tie
    ent_hit=(entropy[one].argmin(1)==good_branch)&~entropy_tie
    ratio=np.maximum(scale[:,0],scale[:,1])/np.maximum(np.minimum(scale[:,0],scale[:,1]),1e-8)
    print(f'  CONFIDENCE among exactly-one-correct N={n_one}: higher Pmax correct={int(conf_hit.sum())}, ties={int(conf_tie.sum())}; '
        f'lower entropy correct={int(ent_hit.sum())}, ties={int(entropy_tie.sum())}. '
        f'RMS larger/smaller median={np.median(ratio):.3f}, p90={np.percentile(ratio,90):.3f}',flush=True)
    # Equal centered logits have identical argmax to equal raw logits. Scaling
    # is the ONLY prediction-changing operation in this counterfactual.
    variants={'equal_logits':np.full((len(y),2),.5)}
    variants.update({m:np.asarray([r['weights'] for r in rows],dtype=float) for m,rows in rows_by_method.items() if m.endswith('_logits')})
    interventions={}
    for method,w in variants.items():
        before=(w[:,:,None]*z).sum(1).argmax(1)==y
        after=(w[:,:,None]*normalized).sum(1).argmax(1)==y
        interventions[method]=(int(before.sum()),int(after.sum()),len(y))
        print(f'  SCALE intervention {method}: original={pct(before):.2f}%, unit-RMS={pct(after):.2f}%, '
            f'fixed errors={int((~before&after).sum())}, new errors={int((before&~after).sum())}. '
            'Same frozen outputs and weights; no fitted temperature.',flush=True)
    for method,rows in rows_by_method.items():
        w=np.asarray([r['weights'] for r in rows],dtype=float)
        a,b=(lu,lh) if method.endswith('_logits') else (p[:,0],p[:,1])
        intervals=[winning_interval(u,h,int(t)) for u,h,t in zip(a,b,y)]
        reachable=np.array([attainable_weight(u,h,int(t))[0] for u,h,t in zip(a,b,y)])
        widths=np.array([v[1]-v[0] if v is not None else np.nan for v in intervals])
        current=np.array([r['pred'] for r in rows])==y
        missed=reachable&~current
        distance=np.array([max(v[0]-weight,weight-v[1],0.) if v is not None else np.nan
            for v,weight in zip(intervals,w[:,0])])
        ties=np.abs(w[one,0]-w[one,1])<=1e-6
        hit=(w[one].argmax(1)==good_branch)&~ties
        print(f'  WEIGHT {method}: reachable={int(reachable.sum())}/{len(y)}, missed={int(missed.sum())}, '
            f'median reachable interval width={np.median(widths[reachable]) if reachable.any() else float("nan"):.4f}, '
            f'narrow(width<0.1)={int((reachable&(widths<.1)).sum())}, '
            f'median missed distance={np.median(distance[missed]) if missed.any() else float("nan"):.4f}; '
            f'correct branch higher weight={int(hit.sum())}/{n_one}, ties={int(ties.sum())}',flush=True)
    return interventions


@torch.no_grad()
def audit_reachability(args):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    raw=(args.data/'manifest.json').read_bytes(); manifest=json.loads(raw)
    protocol=read(args.output/'protocol.json')
    if (protocol.get('evaluation_split')!='val' or protocol.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest()
            or manifest.get('representation')!='legacy_global_three_regions'):
        raise ValueError('Expected matching multi-region validation results')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    diagnose=getattr(args,'diagnose_only',False)
    scenarios=SCENARIOS if diagnose else ['clean']+[f'val_awgn_both_snr{snr}dB' for snr in [-15,-10,-5,0,5,10]]
    expected={r['record_id']:manifest['classes'].index(r['class_name']) for r in manifest['records'] if r['split']=='val'}
    classes=manifest['classes']
    methods=protocol['arguments'].get('gate_modes',['gate_ce','gate_quality'])
    print('READ ONLY reachability: reload exact frozen experts; infer existing validation images; no train/raw/test data or file writes.',flush=True)
    print('Oracle uses true labels to choose a different weight for each record. It is NOT a deployable method or measured model accuracy.',flush=True)
    print('Reachable = some UHF weight in [0,1] gives the true class; analytic crossings, float64, first-class tie rule.',flush=True)
    if diagnose:
        print('DIAGNOSIS: all 19 existing validation conditions; confidence, logit scale, reachable-weight intervals. No new modules or training.',flush=True)
    for seed in protocol['arguments']['seeds']:
        scale_results={}
        initial=read(args.output/f'{methods[0]}_seed{seed}'/'val_clean.json')
        origin=initial['frozen_source']; path=Path(origin['path'])
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        if digest.hexdigest()!=origin['sha256']: raise ValueError('Frozen checkpoint changed')
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['method']!='region_equal' or ckpt['seed']!=seed or ckpt['classes']!=classes:
            raise ValueError('Frozen checkpoint identity mismatch')
        model=build_model('region_equal',ckpt['backbone'],ckpt['feat_dim'],len(classes),False).to(device)
        model.load_state_dict(ckpt['state_dict']); model.requires_grad_(False); model.eval(); del ckpt
        print(f'Checked frozen source seed={seed}, SHA256 matches. Device={device}',flush=True)
        print('condition | N | both wrong | probability equal% | logits equal% | probability reachable N(%) | logits reachable N(%) | probability-only N | logits-only N | neither N | both-wrong rescueable probability/logits | tie-sensitive probability/logits',flush=True)
        for scenario in scenarios:
            ds=Paired(manifest,args.data,'val',scenario); collected={}
            for u,h,y,ids in DataLoader(ds,batch_size=args.image_batch,shuffle=False):
                _,lu=model.uhf(u.to(device)); _,lh=model.hfct(h.to(device))
                pu,ph=lu.softmax(1).cpu().numpy(),lh.softmax(1).cpu().numpy()
                lu,lh=lu.cpu().numpy(),lh.cpu().numpy()
                for i,rid in enumerate(ids): collected[rid]=(lu[i],lh[i],pu[i],ph[i])
            if set(collected)!=set(expected): raise ValueError('Inference parent set mismatch')
            order=sorted(expected); y=np.array([expected[rid] for rid in order])
            lu,lh,pu,ph=[np.array([collected[rid][k] for rid in order],dtype=np.float64) for k in range(4)]
            noisy_rows={}
            for method in methods:
                d=read(args.output/f'{method}_seed{seed}'/f'val_{scenario}.json')
                rows=d['predictions']
                if d.get('frozen_source')!=origin or len(rows)!=len(expected) or {p['record_id']:p['true'] for p in rows}!=expected:
                    raise ValueError('Saved prediction source/parent mismatch')
                rows=sorted(rows,key=lambda p:p['record_id']); saved=np.asarray([p['branch_probabilities'] for p in rows])
                if not np.allclose(saved,np.stack([pu,ph],axis=1),atol=1e-7,rtol=1e-5):
                    raise ValueError('Recomputed branch probabilities differ; stop instead of mixing runs')
                if not np.array_equal(saved.argmax(2),np.stack([pu.argmax(1),ph.argmax(1)],axis=1)):
                    raise ValueError('Branch argmax mismatch after recomputation')
                noisy_rows[method]=rows
            prob=np.array([attainable_weight(a,b,int(t)) for a,b,t in zip(pu,ph,y)])
            logit=np.array([attainable_weight(a,b,int(t)) for a,b,t in zip(lu,lh,y)])
            pr,lr=prob[:,0],logit[:,0]; both_wrong=(pu.argmax(1)!=y)&(ph.argmax(1)!=y)
            pct=lambda count:f'{int(count)}/{len(y)} ({100*count/len(y):.2f}%)'
            print(f'{scenario} | {len(y)} | {int(both_wrong.sum())} | '
                f'{100*np.mean(((pu+ph)/2).argmax(1)==y):.2f} | {100*np.mean(((lu+lh)/2).argmax(1)==y):.2f} | '
                f'{pct(pr.sum())} | {pct(lr.sum())} | {int((pr&~lr).sum())} | {int((lr&~pr).sum())} | '
                f'{int((~pr&~lr).sum())} | {int((both_wrong&pr).sum())}/{int((both_wrong&lr).sum())} | '
                f'{int((pr&~prob[:,1]).sum())}/{int((lr&~logit[:,1]).sum())}',flush=True)
            for method,rows in noisy_rows.items():
                w=np.array([p['weights'] for p in rows],dtype=np.float64)
                if w.shape!=(len(y),2) or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1,atol=1e-5):
                    raise ValueError('Invalid gate weights')
                current=np.array([p['pred'] for p in rows])==y
                # This is an inference intervention, not a separately trained logits gate.
                same_w_logits=(w[:,:1]*lu+w[:,1:]*lh).argmax(1)==y
                reachable=lr if method.endswith('_logits') else pr
                if (current&~reachable).any(): raise ValueError('Reachability contradicts saved gate prediction')
                print(f'  {method}: actual correct={int(current.sum())}/{len(y)}; '
                    f'own-fusion-reachable but missed={int((reachable&~current).sum())}; '
                    f'same weights on LOGITS correct={int(same_w_logits.sum())}/{len(y)}',flush=True)
            if diagnose:
                scale_results[scenario]=diagnosis_arrays(lu,lh,y,noisy_rows)
        if diagnose:
            print('SCALE SUMMARY: validation counterfactual only; not a trained/calibrated method or final-test claim.',flush=True)
            for method in scale_results['clean']:
                for target in ['both','uhf','hfct']:
                    values=[scale_results[s][method] for s in scenarios if s.startswith(f'val_awgn_{target}_')]
                    deltas=np.array([100*(after-before)/n for before,after,n in values])
                    print(f'  {method} | {target} | mean delta={deltas.mean():+.2f} pp | '
                        f'conditions improved/tied/worse={int((deltas>0).sum())}/{int((deltas==0).sum())}/{int((deltas<0).sum())}',flush=True)
                before,after,n=scale_results['clean'][method]
                print(f'  {method} | clean | {100*before/n:.2f}% -> {100*after/n:.2f}%',flush=True)
        del model
    print('Done. Tie-sensitive counts require caution; these are diagnostic attainability ceilings for these fixed experts only.',flush=True)


def main(args):
    if getattr(args,'noise_workers',0)<0: raise ValueError('noise-workers must be nonnegative')
    if getattr(args,'noise_benchmark',0):
        if args.diagnose_only or args.reachability_only or args.audit_only: raise ValueError('Choose one diagnostic mode')
        benchmark_noise(args)
        return
    if args.diagnose_only:
        if args.audit_only or args.reachability_only: raise ValueError('Choose one diagnostic mode')
        audit_reachability(args)
        return
    if args.reachability_only:
        if args.audit_only: raise ValueError('Choose one diagnostic mode')
        audit_reachability(args)
        return
    if args.audit_only:
        audit_saved(args)
        return
    if args.raw_data is None or args.source is None: raise ValueError('Training requires --raw-data and --source')
    if not args.gate_modes or len(set(args.gate_modes))!=len(args.gate_modes): raise ValueError('Unique gate modes required')
    if 'feature_concat' in args.gate_modes and args.gate_modes!=['feature_concat']:
        raise ValueError('Run feature_concat alone; preserve the original gate initialization and existing runs')
    if args.epochs<1 or args.image_batch<1: raise ValueError('Positive epochs and image-batch required')
    if args.batch_size<8 or args.batch_size%2: raise ValueError('Use an even gate batch size >=8')
    if not args.raw_data.is_dir(): raise FileNotFoundError(args.raw_data)
    raw=(args.data/'manifest.json').read_bytes(); manifest=json.loads(raw)
    if manifest.get('representation')!='legacy_global_three_regions': raise ValueError('Expected global+three-region data')
    if len({r['group_id'] for r in manifest['records']})!=len(manifest['records']): raise ValueError('Parent group overlap')
    source=json.loads((args.source/'protocol.json').read_text(encoding='utf-8'))
    if source['manifest_sha256']!=hashlib.sha256(raw).hexdigest(): raise ValueError('Source manifest mismatch')
    if source['arguments'].get('online_noise') or source['arguments'].get('train_noise_bank'):
        raise ValueError('Frozen experts must come from clean-only training')
    # Validate all selected image paths before creating a run directory.
    train=Paired(manifest,args.data,'train')
    val={s:Paired(manifest,args.data,'val',s) for s in SCENARIOS}
    for s in SCENARIOS[1:]:
        config=json.loads((args.data/'images'/s/'config.json').read_text(encoding='utf-8'))
        if config.get('split')!='val': raise ValueError('Wrong validation noise split')
    for seed in args.seeds:
        if not (args.source/f'region_equal_seed{seed}'/'best.pt').is_file(): raise FileNotFoundError(f'Source seed {seed}')
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        manifest_sha256=hashlib.sha256(raw).hexdigest(),evaluation_split='val',source_protocol=source,
        scenarios=SCENARIOS,selection='mean macro F1 across these 19 FIXED validation conditions; first maximum wins',
        gate=('not used: raw modality features concatenated into a linear classification head' if args.gate_modes==['feature_concat'] else
            'normalized branch features + probabilities + max probability/entropy/margin -> 64 -> 2 softmax; identity initialization'),
        fusion={mode:('concatenate raw frozen modality features -> Linear(2*dim, classes)' if mode=='feature_concat' else
            'weighted raw branch logits' if mode.endswith('_logits') else 'weighted branch softmax probabilities') for mode in args.gate_modes},
        baselines='equal_logits and equal_probs use identical frozen experts',
        loss='CE variants: classification CE; quality variants: CE + reliability KL, coefficient 1; gate_interval_logits: CE + mean L1 distance to feasible training-label weight interval, coefficient 1',
        interval_supervision=(dict(target='detached raw-logit class inequalities on train batches only; UHF weight in [0,1]',
            penalty='relu(lower-w_U)+relu(w_U-upper), batch mean; coefficient 1; replaces KL',
            infeasible='zero interval penalty; classification CE retained; zero-width tie-only intervals excluded',
            inference='original gate forward with raw-logit fusion; no labels or interval computation')
            if 'gate_interval_logits' in args.gate_modes else None),
        training='frozen clean-trained region_equal experts; fresh train waveform AWGN uniform [-15,10] dB; only fusion head optimized',
        batch='half clean, half noisy; three noisy targets balanced 5/5/6 at batch32, rotated; sampling with replacement',
        preprocessing_execution='optional bounded spawn CPU workers; parent-planned RNG draws, ordered results, unchanged per-record GPU inference',
        source_code_sha256={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
            for p in ['revision_reliability.py','revision_data.py','revision_train.py']},
        warning='NEW DEVELOPMENT PROTOCOL: noisy validation used for selection; not an independent final test'))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(4)
    results=[]
    for seed in args.seeds:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        path=args.source/f'region_equal_seed{seed}'/'best.pt'
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['method']!='region_equal' or ckpt['seed']!=seed or ckpt['classes']!=manifest['classes']:
            raise ValueError('Source checkpoint identity mismatch')
        dim=ckpt['feat_dim']; classes=len(ckpt['classes'])
        expert=build_model('region_equal',ckpt['backbone'],dim,classes,False).to(device)
        expert.load_state_dict(ckpt['state_dict']); expert.requires_grad_(False); expert.eval()
        del ckpt
        origin=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(f'Caching frozen experts seed={seed}; no test images read.',flush=True)
        clean=cache_existing(expert,train,device,args.image_batch)
        validation={s:cache_existing(expert,ds,device,args.image_batch) for s,ds in val.items()}
        gates={mode:ReliabilityFusion(dim,classes,mode).to(device) for mode in args.gate_modes}
        initial=copy.deepcopy(gates[args.gate_modes[0]].state_dict())
        for gate in gates.values(): gate.load_state_dict(initial)
        optimizers={m:torch.optim.Adam(g.parameters(),lr=.001) for m,g in gates.items()}
        best={m:-1. for m in gates}; states={}; epochs={}; histories={m:[] for m in gates}
        for epoch in range(1,args.epochs+1):
            started=time.perf_counter()
            interval_stats={m:dict(interval_valid=0,interval_total=0,interval_outside=0,interval_distance_sum=0.)
                for m in gates if m=='gate_interval_logits'}
            noise=epoch_noise(expert,clean,manifest,args.raw_data,train.transform,device,seed,epoch,
                workers=getattr(args,'noise_workers',0))
            for g in gates.values(): g.train()
            for u,h,y in balanced_batches(clean,noise,args.batch_size,seed,epoch):
                u,h,y=u.to(device),h.to(device),y.to(device)
                for mode,g in gates.items():
                    optimizers[mode].zero_grad(set_to_none=True)
                    logits,w,lu,lh=g(u,h,return_branches=True)
                    if mode=='gate_interval_logits':
                        penalty,stats=interval_gate_loss(w,lu,lh,y)
                        loss=F.cross_entropy(logits,y)+penalty
                        for key,value in stats.items(): interval_stats[mode][key]+=value
                    elif mode=='feature_concat':
                        loss=F.cross_entropy(logits,y)
                    else:
                        ce,kl=gate_losses(logits,w,lu,lh,y)
                        loss=ce+(kl if mode.startswith('gate_quality') else 0.)
                    loss.backward(); optimizers[mode].step()
            del noise
            for mode,g in gates.items():
                metrics=[evaluate(g,DataLoader(ds,batch_size=64),device,manifest['classes']) for ds in validation.values()]
                score=float(np.mean([v['macro_f1'] for v in metrics]))
                record=dict(epoch=epoch,selection_macro_f1=score,scenario_macro_f1={s:v['macro_f1'] for s,v in zip(SCENARIOS,metrics)})
                extra=''
                if mode in interval_stats:
                    stats=interval_stats[mode]; record['train_interval']=stats
                    valid=stats['interval_valid']; total=stats['interval_total']
                    extra=(f' train_reachable={valid}/{total} train_outside={stats["interval_outside"]}/{valid}'
                        f' interval_loss={stats["interval_distance_sum"]/max(1,total):.4f}')
                record['epoch_seconds']=time.perf_counter()-started
                histories[mode].append(record)
                if score>best[mode]: best[mode]=score; states[mode]=copy.deepcopy(g.state_dict()); epochs[mode]=epoch
                print(f'{mode} seed={seed} epoch={epoch} fixed_val_mean_f1={score:.4f}{extra} epoch_seconds={record["epoch_seconds"]:.1f}',flush=True)
        methods={mode:ReliabilityFusion(dim,classes,mode).to(device) for mode in ['equal_logits','equal_probs']}
        for mode,g in gates.items():
            g.load_state_dict(states[mode]); methods[mode]=g
            folder=args.output/f'{mode}_seed{seed}'
            dump(folder/'history.json',histories[mode])
            torch.save(dict(state_dict=g.state_dict(),dim=dim,classes=manifest['classes'],mode=mode,seed=seed,
                best_epoch=epochs[mode],frozen_source=origin),folder/'best.pt')
        for mode,g in methods.items():
            for scenario,ds in validation.items():
                metrics=evaluate(g,DataLoader(ds,batch_size=64),device,manifest['classes'],diagnostics=True)
                metrics.update(evaluation_split='val',selection_used_this_set=mode in gates,frozen_source=origin)
                dump(args.output/f'{mode}_seed{seed}'/f'val_{scenario}.json',metrics)
                results.append(dict(method=mode,seed=seed,scenario=scenario,evaluation_split='val',
                    **{k:metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','n','correct']}))
                print(f'{mode} seed={seed} {scenario} n={metrics["n"]} acc={metrics["accuracy"]:.4f}',flush=True)
            dump(args.output/'results.json',results)
        del expert,clean,validation,gates,methods,optimizers,states
    print('Done. Expert checkpoints unchanged. Gate checkpoints and validation results saved.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--raw-data',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[3407])
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--image-batch',type=int,default=4)
    p.add_argument('--noise-workers',type=int,default=0,help='Optional CPU waveform/render processes; 0 preserves serial execution. Start with 2; each worker needs waveform RAM.')
    p.add_argument('--noise-benchmark',type=int,default=0,help='Read-only train-waveform benchmark on N noisy parents: compare serial/parallel PNG hashes and time; requires --noise-workers >=1; writes nothing')
    p.add_argument('--gate-modes',nargs='+',choices=GATE_MODES,default=['gate_ce','gate_quality'],help='gate_interval_logits adds training-label weight-interval supervision to logit fusion; original defaults preserved')
    p.add_argument('--audit-only',action='store_true',help='Read saved both-channel predictions only; --output is the existing result directory')
    p.add_argument('--reachability-only',action='store_true',help='No-training probability/logits fusion attainability check on fixed experts and existing validation images')
    p.add_argument('--diagnose-only',action='store_true',help='Read-only confidence/scale/weight-interval diagnosis across all 19 fixed validation conditions; no training or output writes')
    main(p.parse_args())
