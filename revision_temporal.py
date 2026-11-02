"""Temporal window feasibility planning and a causal score penalty.

Planning is read-only: no images, windows, models or intermediate files saved.
Sliding windows retain their original file's split and group. An AC cycle is
not automatically an independent acquisition.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def window_bounds(start,stop,length,stride):
    if length<=0 or stride<=0 or stop<start: raise ValueError('Invalid window configuration')
    return [(i,i+length) for i in range(start,stop-length+1,stride)]


def stability_weights(base_scores,gamma,segments):
    """Causal penalty on normalized log weights, reset at every segment boundary.

    Normalize first so an arbitrary common offset in two gate logits cannot
    create false instability. Only the previous frame of this segment is used.
    This function has no access to labels, SNR or the corruption direction.
    """
    scores=np.asarray(base_scores,dtype=np.float64)
    if scores.ndim!=2 or scores.shape[1]!=2 or len(segments)!=len(scores): raise ValueError('Expected T x 2 scores and T segment IDs')
    if not np.isfinite(scores).all() or not np.isfinite(gamma) or gamma<0: raise ValueError('Finite scores and nonnegative gamma required')
    if not len(scores): return scores.copy(),scores.copy()
    normalized=scores-scores.max(1,keepdims=True)
    normalized-=np.log(np.exp(normalized).sum(1,keepdims=True))
    change=np.zeros_like(normalized)
    for t in range(1,len(scores)):
        if segments[t]==segments[t-1]: change[t]=np.abs(normalized[t]-normalized[t-1])
    adjusted=normalized-gamma*change
    shifted=adjusted-adjusted.max(1,keepdims=True)
    weights=np.exp(shifted); weights/=weights.sum(1,keepdims=True)
    return weights,change


def selected_validation_records(manifest,count):
    parents={}
    for r in manifest['records']:
        if r['split']=='val':
            parent=r.get('parent_record_id',r['record_id'])
            parents.setdefault(parent,dict(r,record_id=parent))
    return [r for cls in manifest['classes'] for r in sorted(
        [r for r in parents.values() if r['class_name']==cls],key=lambda r:r['record_id'])[:count]]


def cycle_pulses(signal,phase,fs):
    from generate_prpd_wfm import legacy_detect_pulse_peaks
    if not np.isfinite(phase).all() or (np.diff(phase)<0).any(): raise ValueError('Expected one complete increasing-phase cycle')
    ph,amp,_=legacy_detect_pulse_peaks(signal,phase,fs,win_us=2.,thr_k=8.)
    indices=np.searchsorted(phase,ph)
    if len(indices) and (indices.max()>=len(phase) or not np.allclose(phase[indices],ph,atol=1e-8)):
        raise ValueError('Cannot map pulse phases to their cycle sample positions')
    return indices,ph,amp


def pilot(args):
    import hashlib
    from io import BytesIO
    import torch
    from PIL import Image
    from torchvision import transforms
    from revision_data import read_parent_record,add_awgn_snr,online_parent_images,REGION_VIEWS
    from revision_train import build_model
    from revision_reliability import ReliabilityFusion
    from generate_prpd_wfm import legacy_compute_phase_from_ch2
    if args.run is None: raise ValueError('Pilot requires --run pointing to the saved frozen-gate experiment')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    raw=(args.data/'manifest.json').read_bytes(); manifest=json.loads(raw)
    protocol=read(args.run/'protocol.json')
    if protocol['manifest_sha256']!=hashlib.sha256(raw).hexdigest() or manifest.get('representation')!='legacy_global_three_regions':
        raise ValueError('Expected matching region manifest and frozen-gate run')
    if protocol.get('evaluation_split')!='val': raise ValueError('Only validation development supported')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    ckpt=torch.load(args.run/f'{args.gate_mode}_seed{args.seed}'/'best.pt',map_location='cpu',weights_only=True)
    if ckpt['seed']!=args.seed or ckpt['mode']!=args.gate_mode or ckpt['classes']!=manifest['classes']:
        raise ValueError('Gate checkpoint mismatch')
    origin=ckpt['frozen_source']; source=Path(origin['path']); digest=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    if digest.hexdigest()!=origin['sha256']: raise ValueError('Frozen source changed')
    expert_ckpt=torch.load(source,map_location='cpu',weights_only=True)
    if expert_ckpt['method']!='region_equal' or expert_ckpt['classes']!=manifest['classes']:
        raise ValueError('Expert checkpoint mismatch')
    expert=build_model('region_equal',expert_ckpt['backbone'],ckpt['dim'],len(ckpt['classes']),False).to(device)
    expert.load_state_dict(expert_ckpt['state_dict']); expert.requires_grad_(False); expert.eval()
    gate=ReliabilityFusion(ckpt['dim'],len(ckpt['classes']),ckpt['mode']).to(device)
    gate.load_state_dict(ckpt['state_dict']); gate.requires_grad_(False); gate.eval()
    del expert_ckpt,ckpt
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    records=selected_validation_records(manifest,args.parents_per_class)
    if not records: raise ValueError('No validation parents')
    print('PILOT ONLY: one middle complete cycle per fixed selected validation file. No training, test access or output files.',flush=True)
    print('Mother boundary=AC cycle. Overlapping windows retain original file group. History resets every cycle/scenario.',flush=True)
    print('Existing experts/gates were trained on 1-second PRPD inputs; short-window inputs are a distribution change. This is feasibility evidence only.',flush=True)
    scenarios=[('clean',False,False),('uhf_noise',True,False),('hfct_noise',False,True),('both_noise',True,True)]
    outcomes={name:{g:[] for g in args.gammas} for name,_,_ in scenarios}
    for record in records:
        waves=read_parent_record(args.raw_data,record); fs=record['sample_rate_hz']
        phase,tz=legacy_compute_phase_from_ch2(waves['phase_ref'].values(),fs,smooth_win=21)
        cycles=[(a,b) for a,b in zip(tz[:-1],tz[1:]) if .015<=b-a<=.025]
        if not cycles: raise ValueError('No complete 15-25ms cycles')
        a,b=cycles[len(cycles)//2]; left,right=int(np.ceil(a*fs)),int(np.ceil(b*fs))
        cycle_phase=phase[left:right].copy(); del phase
        n=right-left
        length=args.window_samples or max(1,round(n*.1))
        stride=args.stride_samples or max(1,round(n*.01))
        bounds=window_bounds(0,n,length,stride)
        if len(bounds)<2: raise ValueError('Need at least two windows wholly inside the cycle')
        if len(bounds)>200 and not args.allow_large:
            raise ValueError(f'{len(bounds)} windows/cycle; use default 10%/1% fractions or explicitly --allow-large after planning workload')
        print(f'Record={record["record_id"]}; class={record["class_name"]}; cycle={left}:{right}; frames={len(bounds)}; window/stride={length}/{stride} points; '
            f'{1000*length/fs:.3f}/{1000*stride/fs:.3f} ms',flush=True)
        features={}; logits={}; empty={}
        for role in ['uhf','hfct']:
            full_clean=waves[role].values()
            seed=int.from_bytes(hashlib.sha256(f'2026/awgn/{record["record_id"]}/{role}'.encode()).digest()[:8],'little')
            full_noisy,quality=add_awgn_snr(full_clean,args.snr_db,np.random.default_rng(seed))
            cyc_clean=full_clean[left:right].copy(); cyc_noisy=full_noisy[left:right].copy()
            local_snr=10*np.log10(np.mean((cyc_clean-np.median(cyc_clean))**2)/np.mean((cyc_noisy-cyc_clean)**2))
            print(f'{role}: SNR referenced to full record={quality["measured_snr_db"]:.6f} dB; measured selected-cycle SNR={local_snr:.3f} dB',flush=True)
            del full_clean,full_noisy
            for contaminated,signal in [(False,cyc_clean),(True,cyc_noisy)]:
                index,ph,amp=cycle_pulses(signal,cycle_phase,fs)
                fu=[]; lu=[]; count_empty=0
                for t,(start,stop) in enumerate(bounds):
                    mask=(index>=start)&(index<stop); count_empty+=int(not mask.any())
                    # Detect once on the mother cycle, then reuse identical pulses in overlapping windows.
                    # Keep absolute phase; do not stretch a short window to an artificial full cycle.
                    encoded=online_parent_images(None,{'record_id':'frame'},manifest['preprocessing']['a_ref'],None,role,0,
                        view_definitions=REGION_VIEWS,pulses_override={role:(ph[mask],amp[mask])})
                    views=[]
                    for view,_,_ in REGION_VIEWS:
                        with Image.open(BytesIO(encoded['frame_'+view][role])) as image:
                            views.append(transform(image.convert('RGB')))
                    x=torch.stack(views).unsqueeze(0).to(device)
                    with torch.no_grad(): f,l=getattr(expert,role)(x)
                    fu.append(f.cpu()); lu.append(l.cpu())
                    if t==0 or (t+1)%20==0 or t+1==len(bounds):
                        print(f'  {role} noisy={contaminated} [{t+1}/{len(bounds)}] RAM only',flush=True)
                features[role,contaminated]=torch.cat(fu); logits[role,contaminated]=torch.cat(lu)
                empty[role,contaminated]=count_empty
            del cyc_clean,cyc_noisy
        del waves
        for name,noisy_u,noisy_h in scenarios:
            u=torch.cat([features['uhf',noisy_u],logits['uhf',noisy_u]],1).to(device)
            h=torch.cat([features['hfct',noisy_h],logits['hfct',noisy_h]],1).to(device)
            with torch.no_grad():
                base=gate.raw_gate_scores(u,h).cpu().numpy()
                original,_,_,_=gate(u,h,return_branches=True)
            lu=logits['uhf',noisy_u].numpy(); lh=logits['hfct',noisy_h].numpy()
            def softmax(z):
                q=np.exp(z-z.max(1,keepdims=True)); return q/q.sum(1,keepdims=True)
            pu,ph=softmax(lu),softmax(lh)
            dc=[float(np.abs(np.diff(p.max(1))).mean()) for p in [pu,ph]]
            target=manifest['classes'].index(record['class_name'])
            print(f'FRAME STABILITY {name}: mean abs confidence jump U/H={dc[0]:.5f}/{dc[1]:.5f}; '
                f'empty-pulse frames U/H={empty["uhf",noisy_u]}/{empty["hfct",noisy_h]} of {len(bounds)}',flush=True)
            for gamma in args.gammas:
                # A fresh segment ID sequence is passed for each mother cycle: no cross-cycle state.
                w,change=stability_weights(base,gamma,[0]*len(bounds))
                z=(w[:,:1]*lu+w[:,1:]*lh) if args.gate_mode.endswith('_logits') else np.log(np.maximum(w[:,:1]*pu+w[:,1:]*ph,1e-8))
                p=softmax(z); prediction=int(p.mean(0).argmax())
                if gamma==0 and not np.allclose(p,original.softmax(1).cpu().numpy(),atol=2e-6):
                    raise ValueError('gamma=0 does not reproduce the base gate')
                outcomes[name][gamma].append(prediction==target)
                print(f'PILOT {name} gamma={gamma:g}: cycle_pred={manifest["classes"][prediction]}; '
                    f'frame agreement with file label={100*np.mean(p.argmax(1)==target):.2f}%; '
                    f'mean weight U/H={w[:,0].mean():.4f}/{w[:,1].mean():.4f}; '
                    f'mean score jumps U/H={change[:,0].mean():.4f}/{change[:,1].mean():.4f}',flush=True)
    print('SMALL FEASIBILITY SUBSET ONLY: one cycle per selected file; not independent window-level accuracy or a final benchmark.',flush=True)
    for scenario,variants in outcomes.items():
        for gamma,correct in variants.items(): print(f'{scenario} gamma={gamma:g}: cycles correct {sum(correct)}/{len(correct)}',flush=True)
    print('Done. All parameters frozen, same per-record noise realization shared by overlapping windows, no files written.',flush=True)


def main(args):
    if args.pilot:
        pilot(args)
        return
    from revision_data import read_parent_record
    from generate_prpd_wfm import legacy_compute_phase_from_ch2
    manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    parents={}
    for r in manifest['records']:
        if r['split']!='val': continue
        parent=r.get('parent_record_id',r['record_id'])
        parents.setdefault(parent,dict(r,record_id=parent))
    selected=[]
    for cls in manifest['classes']:
        selected.extend(sorted([r for r in parents.values() if r['class_name']==cls],key=lambda r:r['record_id'])[:args.parents_per_class])
    if not selected: raise ValueError('No validation records for planning')
    print('READ ONLY planning: select fixed first records per class; no image generation or model inference.',flush=True)
    print('All cycles and windows inherit original file group and split; no new independence assumption.',flush=True)
    print('record | fs Hz | full cycles | median points/cycle | setting | window/stride points | window duration ms | overlap% | cycle-bounded frames/file | 30deg-bounded frames/file',flush=True)
    estimates=[]
    for record in selected:
        waves=read_parent_record(args.raw_data,record); fs=record['sample_rate_hz']
        phase,tz=legacy_compute_phase_from_ch2(waves['phase_ref'].values(),fs,smooth_win=21)
        del phase
        periods=[(a,b) for a,b in zip(tz[:-1],tz[1:]) if .015<=b-a<=.025]
        if not periods: raise ValueError('No valid 15-25 ms cycles')
        sizes=[int(np.ceil(b*fs))-int(np.ceil(a*fs)) for a,b in periods]
        median=int(round(float(np.median(sizes))))
        settings=[('specified',args.window_samples or 2000,args.stride_samples or 200),
                  ('10%-cycle/1%-cycle',max(1,round(median*.1)),max(1,round(median*.01)))]
        for name,length,stride in settings:
            full=local=0
            for a,b in periods:
                edges=np.ceil(np.linspace(a,b,13)*fs).astype(np.int64)
                full+=len(window_bounds(int(edges[0]),int(edges[-1]),length,stride))
                local+=sum(len(window_bounds(int(x),int(y),length,stride)) for x,y in zip(edges[:-1],edges[1:]))
            overlap=max(0.,100*(1-stride/length))
            print(f'{record["record_id"]} | {fs:g} | {len(periods)} | {median} | {name} | '
                f'{length}/{stride} | {1000*length/fs:.6f} | {overlap:.1f} | {full} | {local}',flush=True)
            estimates.append((name,full,local))
        del waves
    for name in ['specified','10%-cycle/1%-cycle']:
        subset=[v for v in estimates if v[0]==name]
        frames=float(np.mean([v[1] for v in subset]))*len(parents)*19
        print(f'ROUGH cycle-bound workload ({name}): {frames:,.0f} frame evaluations across {len(parents)} val files x 19 conditions; '
            f'{frames*8:,.0f} view evaluations, based on this small planning subset.',flush=True)
    print('Full-cycle 10% windows cannot fit inside a 30-degree block (8.33% of a cycle); zero local frames is expected for that setting.',flush=True)
    print('This plans workload only; no temporal model experiment has been run and no files were written.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--raw-data',type=Path,required=True)
    p.add_argument('--window-samples',type=int,help='Explicit size; pilot default is 10% of measured cycle')
    p.add_argument('--stride-samples',type=int,help='Explicit stride; pilot default is 1% of measured cycle')
    p.add_argument('--parents-per-class',type=int,default=1)
    p.add_argument('--pilot',action='store_true',help='Frozen-gate feasibility run on one middle cycle per selected validation file; no files written')
    p.add_argument('--run',type=Path,help='Saved frozen reliability gate experiment directory')
    p.add_argument('--gate-mode',choices=['gate_ce','gate_quality','gate_ce_logits','gate_quality_logits'],default='gate_ce_logits')
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--snr-db',type=float,default=-15.)
    p.add_argument('--gammas',type=float,nargs='+',default=[0.,1.])
    p.add_argument('--allow-large',action='store_true',help='Explicitly allow more than 200 windows/cycle in a pilot')
    args=p.parse_args()
    if ((args.window_samples is not None and args.window_samples<1) or (args.stride_samples is not None and args.stride_samples<1)
            or args.parents_per_class<1): p.error('Positive window, stride and parent count required')
    if not np.isfinite(args.snr_db) or not -100<=args.snr_db<=100: p.error('SNR must be finite -100..100 dB')
    if any(not np.isfinite(g) or g<0 for g in args.gammas) or len(set(args.gammas))!=len(args.gammas) or 0. not in args.gammas:
        p.error('Use unique finite nonnegative gammas including 0 as control')
    main(args)
