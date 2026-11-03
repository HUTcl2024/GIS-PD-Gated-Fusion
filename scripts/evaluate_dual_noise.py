"""Evaluate saved concat baselines and reuse GFM predictions; no training.

Outputs one CSV per pollution direction, ordered by SNR then method.
Requires the same frozen manifest and the existing 159-record test images.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_train import Paired
from train_dual_concat import DualConcat, evaluate, pooled, print_result
from svm_prpd import validate, metrics, save


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svm_scores(bundle, data, scenario, records, batch_size=512):
    scores = []
    for start in range(0, len(records), batch_size):
        parts = []
        for role, pca in zip(['uhf', 'hfct'], bundle['pca']):
            x = []
            for r in records[start:start+batch_size]:
                path = data/'images'/scenario/role/r['class_name']/(r['record_id']+'.png')
                with Image.open(path) as image:
                    x.append(np.asarray(image.convert('L').resize((64, 64), Image.Resampling.BILINEAR),
                                        dtype=np.float32).reshape(-1)/255.)
            parts.append(pca.transform(np.stack(x)))
        features = bundle['scaler'].transform(np.concatenate(parts, axis=1))
        scores.append(bundle['svm'].decision_function(features))
    return np.concatenate(scores)


def main(args):
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new directory')
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU requires --device cpu')
    manifest = read(args.data/'manifest.json')
    tiled, counts = validate(manifest)
    if not tiled or manifest.get('split_seed') != 42:
        raise ValueError('Expected archived split-42 12-tile data')
    digest = sha(args.data/'manifest.json')
    classes = manifest['classes']
    records = [r for r in manifest['records'] if r['split'] == 'test']
    parents = {r['parent_record_id']: classes.index(r['class_name']) for r in records}
    sources = {name: args.runs/f'dual_{name}_concat_split42_seed3407' for name in ['svm', 'cnn', 'resnet50']}
    source_files = {}
    for name, folder in sources.items():
        protocol = read(folder/'protocol.json')
        if protocol['manifest_sha256'] != digest or protocol['arguments']['seed'] != 3407:
            raise ValueError('Baseline manifest/seed mismatch: ' + name)
        if protocol['arguments']['method'] != name:
            raise ValueError('Baseline identity mismatch: ' + name)
        source_files[name] = folder/('best.joblib' if name == 'svm' else 'best.pt')
    if read(args.gfm_results/'protocol.json')['manifest_sha256'] != digest:
        raise ValueError('GFM manifest mismatch')
    scenarios = [f'awgn_{target}_snr{snr}dB' for target in args.targets for snr in [-15, -10, -5]]
    datasets = {}
    gfm = {}
    # Full preflight before creating any output or loading models.
    for scenario in scenarios:
        config = read(args.data/'images'/scenario/'config.json')
        if config.get('split') != 'test' or config.get('noise') != 'AWGN':
            raise ValueError('Expected fixed test AWGN')
        datasets[scenario] = Paired(manifest, args.data, 'test', scenario)
        path = args.gfm_results/'gfm_seed3407'/f'test_{scenario}.json'
        source_files['gfm_'+scenario] = path
        d = read(path)
        rows = d['predictions']
        if (d['n'] != 159 or len(rows) != 159 or len({r['record_id'] for r in rows}) != 159
                or {r['record_id']: r['true'] for r in rows} != parents):
            raise ValueError('GFM test predictions do not match manifest parents/labels')
        m = metrics([r['true'] for r in rows], [r['pred'] for r in rows], classes)
        if not np.isclose(m['accuracy'], d['accuracy']) or not np.isclose(m['f1_macro'], d['macro_f1']):
            raise ValueError('Saved GFM metrics do not match predictions')
        m.update(predictions=rows, metric_unit='original_record', source=str(path))
        gfm[scenario] = m
    before = {name: sha(path) for name, path in source_files.items()}
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', dict(manifest_sha256=digest, split_seed=42, seed=3407,
        evaluation_split='test', scenarios=scenarios, source_files={k: str(v) for k, v in source_files.items()},
        source_sha256=before, selection='Existing fixed checkpoints; no training, tuning or checkpoint selection',
        script_sha256=sha(Path(__file__))))
    results = {}
    for name in ['svm', 'cnn', 'resnet50', 'gfm']:
        if name == 'svm':
            # Load only a trusted local joblib written by train_dual_concat.py.
            model = joblib.load(source_files[name])
            if model['classes'] != classes:
                raise ValueError('SVM class order mismatch')
        elif name != 'gfm':
            c = torch.load(source_files[name], map_location='cpu', weights_only=True)
            if c['method'] != name or c['classes'] != classes:
                raise ValueError('Checkpoint identity mismatch')
            model = DualConcat(name, len(classes), False).to(device)
            model.load_state_dict(c['state_dict'])
        for scenario in scenarios:
            if name == 'svm':
                m = pooled(svm_scores(model, args.data, scenario, records), records, classes, False)
            elif name == 'gfm':
                m = gfm[scenario]
            else:
                m = evaluate(model, DataLoader(datasets[scenario], batch_size=args.batch_size), device, classes)
            if m['n'] != 159:
                raise ValueError('Expected record-level n=159')
            results[name, scenario] = m
            save(args.output/name/f'test_{scenario}.json', m)
            print_result(name+' '+scenario, m)
        if name != 'gfm':
            del model
    table = []
    for target in args.targets:
        rows = []
        for snr in [-15, -10, -5]:
            scenario = f'awgn_{target}_snr{snr}dB'
            for name in ['svm', 'cnn', 'resnet50', 'gfm']:
                m = results[name, scenario]
                rows.append(dict(pollution=target, snr_db=snr, method=name, n=159,
                    accuracy_pct=100*m['accuracy'], precision_macro_pct=100*m['precision_macro'],
                    f1_macro_pct=100*m['f1_macro']))
        with (args.output/f'table_{target}.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        table.extend(rows)
        print('\nPollution=' + target + ' | SNR dB | Method | Accuracy% | Precision macro% | F1 macro%', flush=True)
        for row in rows:
            print(f"{target} | {row['snr_db']} | {row['method']} | {row['accuracy_pct']:.2f} | "
                  f"{row['precision_macro_pct']:.2f} | {row['f1_macro_pct']:.2f}", flush=True)
    save(args.output/'summary.json', table)
    if any(sha(source_files[name]) != digest for name, digest in before.items()):
        raise ValueError('A source changed during evaluation')
    print(f'Complete: {len(table)} results, all n=159; CSV ordered by SNR then method. Source files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--gfm-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', choices=['both', 'uhf', 'hfct'], nargs='+', default=['both', 'uhf', 'hfct'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    main(parser.parse_args())
