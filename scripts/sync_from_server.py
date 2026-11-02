"""Synchronize an explicit code allowlist from the server into this checkout.

Default: inspect hashes only. No raw data, models or recursive directory copies.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FILES=(
    'revision_data.py','revision_train.py','revision_reliability.py',
    'generate_prpd_wfm.py','mso44_wfm.py',
    'mso44_csv.py','recovered_models.py',
)
OPTIONAL_FILES=('revision_temporal.py',)
REQUIRED={
    'revision_data.py':{'dump','image_paths','region_paths'},
    'revision_train.py':{'Paired','build_model','evaluate'},
    'revision_reliability.py':{'ReliabilityFusion'},
}
ROOT=Path(__file__).resolve().parents[1]
REPO='HUTcl2024/GIS-PD-Gated-Fusion'

def git(*args,capture=True):
    return subprocess.run(['git','-C',str(ROOT),*args],check=True,text=True,
        stdout=subprocess.PIPE if capture else None).stdout

def validated_source(source):
    if source.resolve()==ROOT: raise ValueError('Source must be the separate experiment directory')
    payload={}
    for name in FILES+OPTIONAL_FILES:
        path=source/name
        if name in OPTIONAL_FILES and not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file(): raise ValueError('Missing or symlink source: '+name)
        raw=path.read_bytes()
        if len(raw)>1024*1024: raise ValueError('Unexpectedly large source: '+name)
        tree=ast.parse(raw.decode('utf-8-sig'),filename=name)
        symbols={n.name for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))}
        missing=REQUIRED.get(name,set())-symbols
        if missing:
            raise ValueError(name+' lacks core definitions: '+', '.join(sorted(missing))+
                '; check file identity before syncing (new experimental features are not required)')
        if any(isinstance(n,ast.ImportFrom) and n.module==path.stem for n in ast.walk(tree)):
            raise ValueError(name+' imports itself; possible wrong-file upload')
        payload[name]=raw
    return payload

def feature_report(payload):
    """Report optional version features without requiring or executing them."""
    trees={name:ast.parse(raw.decode('utf-8-sig')) for name,raw in payload.items()}
    symbols={name:{n.name for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))}
        for name,tree in trees.items()}
    constants={n.value for n in ast.walk(trees['revision_data.py']) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    return dict(region_resplit='region_resplit' in symbols['revision_data.py'],
        explicit_split_seed='--split-seed' in constants,
        interval_supervision='interval_gate_loss' in symbols['revision_reliability.py'],
        parallel_noise_preparation='ordered_noise_images' in symbols['revision_reliability.py'],
        temporal_script='revision_temporal.py' in payload)

def main(args):
    if (args.commit or args.push) and not args.apply: raise ValueError('--commit/--push require --apply')
    if args.push and not args.commit: raise ValueError('--push requires --commit')
    payload=validated_source(args.source)
    capabilities=feature_report(payload)
    print('Detected server features: '+json.dumps(capabilities,sort_keys=True))
    optional_missing=[name for name in OPTIONAL_FILES if name not in payload]
    if optional_missing:
        print('Optional files absent on server (not an error): '+', '.join(optional_missing))
    changed=[]
    for name,raw in payload.items():
        digest=hashlib.sha256(raw).hexdigest()
        normalized=raw.replace(b'\r\n',b'\n')
        same=(ROOT/name).is_file() and (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==normalized
        print(f'{name}: {"same LF content" if same else "different"} source_sha256={digest}')
        if not same: changed.append(name)
    if not args.apply:
        print('Read only. To copy code and update provenance, use --apply; optionally --commit --push.')
        return
    top=Path(git('rev-parse','--show-toplevel').strip()).resolve()
    if top!=ROOT: raise ValueError('Run this from its own repository checkout')
    remote=git('remote','get-url','origin').strip()
    if remote not in {f'https://github.com/{REPO}.git',f'https://github.com/{REPO}',f'git@github.com:{REPO}.git'}:
        raise ValueError('Unexpected origin; no files changed')
    if git('status','--porcelain').strip(): raise ValueError('Checkout has uncommitted changes; no files changed')
    branch=git('branch','--show-current').strip()
    if not branch: raise ValueError('Detached HEAD; no files changed')
    for name,raw in payload.items():
        if (ROOT/name).is_symlink(): raise ValueError('Symlink destination: '+name)
    for name in changed: (ROOT/name).write_bytes(payload[name].replace(b'\r\n',b'\n'))
    provenance=dict(snapshot='server_code_copy',server_verified=True,
        verification_scope='Only files listed under files were read from the server; no experiment results were verified',
        capabilities=capabilities,
        optional_files_not_copied=optional_missing,
        preserved_local_optional={name:dict(sha256=hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            reason='Kept repository version; absent on server and not server-verified')
            for name in optional_missing if (ROOT/name).is_file()},
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        line_endings='Repository code uses LF; source_sha256 hashes original bytes, sha256 hashes LF-normalized repository bytes',
        files={n:dict(sha256=hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest(),
            source_sha256=hashlib.sha256(b).hexdigest(),bytes=len(b.replace(b'\r\n',b'\n'))) for n,b in payload.items()},
        exclusions='No data, manifests, checkpoints, predictions, logs or manuscripts copied')
    (ROOT/'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Copied only the code allowlist; updated SOURCE_PROVENANCE.json.')
    if args.commit:
        git('add','--',*payload,'SOURCE_PROVENANCE.json')
        if subprocess.run(['git','-C',str(ROOT),'diff','--cached','--quiet'],check=False).returncode:
            git('-c','user.name=HUTcl2024','-c','user.email=170594196+HUTcl2024@users.noreply.github.com',
                'commit','-m','Sync experiment source code from server',capture=False)
        else: print('No staged difference; no new commit needed.')
    if args.push: git('push','origin',branch,capture=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--commit',action='store_true')
    p.add_argument('--push',action='store_true')
    main(p.parse_args())
