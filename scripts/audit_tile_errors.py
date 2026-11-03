"""Read saved tile-model predictions and export record-level error diagnostics.

No inference, training, raw data access or checkpoint changes. Mean tile weights
and averaged branch probabilities are summaries; they do not establish the
causal effect of gating. Internal branch heads are not independent baselines.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    if args.output.exists():
        raise FileExistsError('Use a new output directory; no overwrite')
    manifest = read(args.data/'manifest.json')
    if manifest.get('representation') != 'legacy_phase_tiles_30deg':
        raise ValueError('Expected a 12-tile manifest')
    classes = manifest['classes']
    index = {r['parent_record_id']: classes.index(r['class_name']) for r in manifest['records'] if r['split'] == args.split}
    protocol = read(args.results/'protocol.json')
    digest = hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    if protocol['manifest_sha256'] != digest:
        raise ValueError('Result manifest does not match data')
    raw_rows = read(args.results/'results.json')
    datasets = {}
    for row in raw_rows:
        if row.get('evaluation_split', args.split) != args.split:
            raise ValueError('Mixed evaluation splits')
        method, seed, scenario = row['method'], row['seed'], row['scenario']
        key = method, seed, scenario
        if key in datasets:
            raise ValueError('Duplicate result key')
        path = args.results/f'{method}_seed{seed}'/f'{args.split}_{scenario}.json'
        d = read(path)
        predictions = d['predictions']
        if (d['n'] != len(index) or len(predictions) != len(index)
                or {r['record_id']: r['true'] for r in predictions} != index):
            raise ValueError('Prediction IDs/labels/count differ from manifest')
        datasets[key] = {r['record_id']: r for r in predictions}
    errors, summaries, class_rows, confusion = [], [], [], {}
    for (method, seed, scenario), predictions in sorted(datasets.items()):
        matrix = [[0]*len(classes) for _ in classes]
        partitions = Counter()
        for parent, r in predictions.items():
            truth, pred = r['true'], r['pred']
            matrix[truth][pred] += 1
            entry = dict(method=method, seed=seed, scenario=scenario, record_id=parent,
                         true_class=classes[truth], predicted_class=classes[pred])
            w = r.get('weights')
            if w is not None:
                entry.update(mean_tile_uhf_weight=w[0], mean_tile_hfct_weight=w[1])
            bp = r.get('branch_predictions')
            if bp is not None:
                u, h = bp[0] == truth, bp[1] == truth
                group = 'both_correct' if u and h else 'only_uhf_correct' if u else 'only_hfct_correct' if h else 'both_wrong'
                partitions[group] += 1
                entry.update(internal_uhf_prediction=classes[bp[0]], internal_hfct_prediction=classes[bp[1]], branch_group=group)
                if w is not None and u != h:
                    entry['correct_branch_has_larger_mean_weight'] = (w[0] > w[1]) if u else (w[1] > w[0])
            for baseline in ['concat', 'equal', 'gfm_no_conf', 'uhf', 'hfct']:
                other = datasets.get((baseline, seed, scenario), {}).get(parent)
                if other is not None:
                    entry[baseline+'_prediction'] = classes[other['pred']]
                    entry[baseline+'_correct'] = other['pred'] == truth
            if pred != truth:
                errors.append(entry)
        correct = sum(matrix[i][i] for i in range(len(classes)))
        summaries.append(dict(method=method, seed=seed, scenario=scenario, n=len(index),
            correct=correct, errors=len(index)-correct, accuracy_pct=100*correct/len(index), **partitions))
        confusion[f'{method}/seed{seed}/{scenario}'] = matrix
        for i, label in enumerate(classes):
            support = sum(matrix[i])
            class_rows.append(dict(method=method, seed=seed, scenario=scenario, class_name=label,
                support=support, correct=matrix[i][i], recall_pct=100*matrix[i][i]/support if support else None))
    args.output.mkdir(parents=True, exist_ok=False)
    error_columns = ['method','seed','scenario','record_id','true_class','predicted_class',
        'mean_tile_uhf_weight','mean_tile_hfct_weight','internal_uhf_prediction','internal_hfct_prediction',
        'branch_group','correct_branch_has_larger_mean_weight']
    for baseline in ['concat', 'equal', 'gfm_no_conf', 'uhf', 'hfct']:
        error_columns += [baseline+'_prediction', baseline+'_correct']
    save_csv(args.output/'error_records.csv', errors, error_columns)
    save_csv(args.output/'gate_errors.csv', [r for r in errors if r['method'] in ['gfm','cnn_gfm']], error_columns)
    save_csv(args.output/'summary.csv', summaries, ['method','seed','scenario','n','correct','errors','accuracy_pct',
        'both_correct','only_uhf_correct','only_hfct_correct','both_wrong'])
    save_csv(args.output/'class_recall.csv', class_rows, ['method','seed','scenario','class_name','support','correct','recall_pct'])
    (args.output/'confusion_matrices.json').write_text(json.dumps(dict(classes=classes, matrices=confusion), indent=2), encoding='utf-8')
    (args.output/'README.txt').write_text(
        f'Evaluation split: {args.split}. Manifest SHA256: {digest}\n'
        'Each CSV row is an original record, not a phase tile.\n'
        'Weights are means across 12 tiles. Branch predictions aggregate internal branch probabilities.\n'
        'Internal branches differ from separately trained single-channel baselines.\n'
        'Missing branch fields mean the old prediction export did not save them.\n'
        'Correct branch having larger mean weight does not guarantee a correct fused result.\n'
        'Comparisons describe saved predictions, not causal proof.\n', encoding='utf-8')
    print(f'Completed: {len(datasets)} conditions/models, {len(errors)} error rows.')
    for row in summaries:
        if row['method'] in ['gfm','cnn_gfm']:
            print(f"{row['method']} | {row['scenario']} | errors={row['errors']}/{row['n']}")
    print('Saved:', args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val','test'], required=True)
    main(parser.parse_args())
