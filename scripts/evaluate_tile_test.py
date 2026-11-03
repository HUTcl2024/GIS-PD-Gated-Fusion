"""Evaluate fixed 12-tile checkpoints; no training, selection or raw waveform access.

Writes a NEW result directory. Existing checkpoints and development results stay
unchanged. A rerun must use a different output directory; there is no resume.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['uhf', 'hfct', 'concat', 'equal', 'gfm', 'gfm_no_conf']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def preflight(args):
    if args.output.exists():
        raise FileExistsError('Output already exists; do not overwrite: ' + str(args.output))
    if not args.methods or len(set(args.methods)) != len(args.methods):
        raise ValueError('Choose unique methods')
    if not args.snrs or len(set(args.snrs)) != len(args.snrs):
        raise ValueError('Choose unique SNRs')
    if args.batch_size < 1 or args.threads < 1 or args.expected_records < 1:
        raise ValueError('Positive batch size, thread count and record count required')
    manifest_path = args.data/'manifest.json'
    manifest = read(manifest_path)
    protocol = read(args.source/'protocol.json')
    if manifest.get('representation') != 'legacy_phase_tiles_30deg':
        raise ValueError('Expected the original 12 x 30-degree tile representation')
    if protocol.get('manifest_sha256') != digest(manifest_path):
        raise ValueError('Manifest differs from the one used for training')
    if manifest.get('split_seed') != args.expected_split_seed:
        raise ValueError('Unexpected data split seed')
    settings = protocol['arguments']
    if any(settings.get(k, False) for k in ['smoke', 'online_noise', 'train_noise_bank', 'two_stage']):
        raise ValueError('This entry point is for full clean-trained tile models only')
    if settings.get('aux_weight', 0) != 0:
        raise ValueError('The paper tile experiment did not use auxiliary supervision')
    parents = {}
    for record in manifest['records']:
        parents.setdefault(record['parent_record_id'], []).append(record)
    for children in parents.values():
        if (len(children) != 12 or len({r['split'] for r in children}) != 1
                or len({r['class_name'] for r in children}) != 1
                or {r['phase_start_deg'] for r in children} != set(range(0, 360, 30))):
            raise ValueError('A parent must have 12 phase tiles in exactly one split')
    counts = Counter(children[0]['split'] for children in parents.values())
    if counts['test'] != args.expected_records:
        raise ValueError('Unexpected test parent count: ' + str(counts))
    scenarios = ['clean'] + [f'awgn_{target}_snr{snr}dB'
        for target in ['both', 'uhf', 'hfct'] for snr in args.snrs]
    for target in ['both', 'uhf', 'hfct']:
        for snr in args.snrs:
            config = read(args.data/'images'/f'awgn_{target}_snr{snr}dB'/'config.json')
            if (config.get('split') != 'test' or config.get('target') != target
                    or config.get('snr_db') != snr or config.get('noise') != 'AWGN'):
                raise ValueError('Test noise configuration mismatch')
    checkpoints = {method: args.source/f'{method}_seed{args.seed}'/'best.pt'
                   for method in args.methods}
    hashes = {method: digest(path) for method, path in checkpoints.items()}
    return manifest, protocol, scenarios, checkpoints, hashes


def main(args):
    manifest, protocol, scenarios, checkpoints, hashes = preflight(args)
    # Lazy imports keep --help usable without installing PyTorch.
    sys.path.insert(0, str(ROOT))
    import torch
    from torch.utils.data import DataLoader
    from revision_train import Paired, build_model, evaluate

    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu explicitly if intended')
    device = torch.device(args.device)
    for scenario in scenarios:
        dataset = Paired(manifest, args.data, 'test', scenario)
        if len(dataset) != args.expected_records * 12:
            raise ValueError('Incomplete test tile dataset: ' + scenario)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', dict(
        checkpoint_source=str(args.source.resolve()), source_protocol=protocol,
        manifest_sha256=digest(args.data/'manifest.json'),
        split_seed=manifest['split_seed'], training_seed=args.seed,
        evaluation_split='test', methods=args.methods, scenarios=scenarios,
        expected_records=args.expected_records, checkpoint_sha256=hashes,
        selection='Unchanged source checkpoints; no retraining or test-driven selection',
        pooling='Arithmetic mean of 12 tile softmax probability vectors per parent',
        source_code_sha256={name: digest(ROOT/name) for name in
            ['scripts/evaluate_tile_test.py', 'revision_train.py', 'revision_data.py']},
        torch=torch.__version__, device=str(device), batch_size=args.batch_size,
        warning='Same record partition was inspected in earlier development experiments; not an external untouched test set'))
    rows = []
    print('Evaluation only: fixed checkpoints, test records, no optimization.', flush=True)
    for method, path in checkpoints.items():
        if digest(path) != hashes[method]:
            raise ValueError('Checkpoint changed after preflight')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint['method'] != method or checkpoint['seed'] != args.seed
                or checkpoint['classes'] != manifest['classes']):
            raise ValueError('Checkpoint identity mismatch')
        model = build_model(method, checkpoint['backbone'], checkpoint['feat_dim'],
                            len(manifest['classes']), False).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        for scenario in scenarios:
            dataset = Paired(manifest, args.data, 'test', scenario)
            metrics = evaluate(model, DataLoader(dataset, batch_size=args.batch_size),
                               device, manifest['classes'])
            if metrics['n'] != args.expected_records or metrics.get('metric_unit') != 'original_record':
                raise ValueError('Expected original-record metrics')
            save(args.output/f'{method}_seed{args.seed}'/f'test_{scenario}.json', metrics)
            rows.append(dict(method=method, seed=args.seed, scenario=scenario,
                evaluation_split='test', metric_unit='original_record',
                checkpoint_sha256=hashes[method],
                **{k: metrics[k] for k in ['n', 'correct', 'accuracy', 'macro_f1', 'balanced_accuracy']}))
            save(args.output/'results.json', rows)
            print(f"{method} | {scenario} | n={metrics['n']} | "
                  f"Acc={100*metrics['accuracy']:.2f}% | F1={100*metrics['macro_f1']:.2f}%", flush=True)
        del model, checkpoint
        if digest(path) != hashes[method]:
            raise ValueError('Source checkpoint changed during evaluation')
    print(f'Complete: {len(rows)} results. Source checkpoints unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--expected-split-seed', type=int, default=42)
    parser.add_argument('--expected-records', type=int, default=159)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--snrs', nargs='+', type=int, default=[-15, -10, -5])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    main(parser.parse_args())
