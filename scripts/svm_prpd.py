"""Clean PRPD baseline: flattened pixels -> train-only PCA/scaling -> RBF SVM.

Uses the existing manifest; never resplits records. Fixed hyperparameters, no
test tuning. For 12-tile inputs, average class decision scores per parent.
Scores are not calibrated probabilities. Fits separate UHF/HFCT classifiers.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def validate(manifest):
    records = manifest['records']
    tiled = manifest.get('representation') == 'legacy_phase_tiles_30deg'
    if manifest.get('representation') not in (None, 'whole_record_prpd', 'legacy_phase_tiles_30deg'):
        raise ValueError('Use the paper whole-image or 12-tile manifest, not global/region views')
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Duplicate image IDs')
    parents = {}
    for r in records:
        parent = r['parent_record_id'] if tiled else r['record_id']
        parents.setdefault(parent, []).append(r)
    for children in parents.values():
        if (len({r['split'] for r in children}) != 1
                or len({r['class_name'] for r in children}) != 1):
            raise ValueError('A parent crosses splits or labels')
        if tiled and (len(children) != 12 or
                      {r['phase_start_deg'] for r in children} != set(range(0, 360, 30))):
            raise ValueError('Each parent requires 12 distinct phase tiles')
    counts = Counter(children[0]['split'] for children in parents.values())
    if counts != {'train': 474, 'val': 158, 'test': 159}:
        raise ValueError('Expected 474/158/159 original records; found ' + str(counts))
    return tiled, counts


def load_features(records, data, channel, size, classes):
    x = np.empty((len(records), size * size), dtype=np.float32)
    y = np.empty(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        path = data/'images'/'clean'/channel/r['class_name']/(r['record_id']+'.png')
        with Image.open(path) as image:
            image = image.convert('L').resize((size, size), Image.Resampling.BILINEAR)
            x[i] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.
        y[i] = classes.index(r['class_name'])
    return x, y


def metrics(truth, pred, classes):
    labels = list(range(len(classes)))
    return dict(n=len(truth), correct=int(np.sum(np.asarray(truth) == np.asarray(pred))),
        accuracy=float(accuracy_score(truth, pred)),
        precision_macro=float(precision_score(truth, pred, labels=labels, average='macro', zero_division=0)),
        recall_macro=float(recall_score(truth, pred, labels=labels, average='macro', zero_division=0)),
        f1_macro=float(f1_score(truth, pred, labels=labels, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(truth, pred, labels=labels).tolist(),
        classification_report=classification_report(truth, pred, labels=labels,
            target_names=classes, output_dict=True, zero_division=0))


def evaluate(model, x, records, classes, tiled):
    scores = model.decision_function(x)
    groups = {}
    for score, r in zip(scores, records):
        parent = r['parent_record_id'] if tiled else r['record_id']
        groups.setdefault(parent, []).append((score, r))
    rows = []
    for parent, items in groups.items():
        score = np.mean([s for s, _ in items], axis=0)
        rows.append(dict(record_id=parent, true=classes.index(items[0][1]['class_name']),
                         pred=int(score.argmax()), decision_scores=score.tolist(), blocks=len(items)))
    result = metrics([r['true'] for r in rows], [r['pred'] for r in rows], classes)
    result.update(metric_unit='original_record', predictions=rows,
                  pooling='mean of per-tile one-vs-rest SVM decision scores' if tiled else 'single image')
    if tiled:
        result['tile_level'] = metrics([classes.index(r['class_name']) for r in records],
                                       scores.argmax(axis=1), classes)
    return result


def main(args):
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing results will not be overwritten')
    if args.size < 8 or args.pca < 1 or args.c <= 0:
        raise ValueError('Use size>=8, PCA>=1 and C>0')
    raw = (args.data/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    tiled, counts = validate(manifest)
    classes = manifest['classes']
    if len(classes) != 4:
        raise ValueError('Expected four classes')
    train = [r for r in manifest['records'] if r['split'] == 'train']
    test = [r for r in manifest['records'] if r['split'] == 'test']
    # Check all required paths before fitting either classifier.
    for channel in ['uhf', 'hfct']:
        for r in train + test:
            p = args.data/'images'/'clean'/channel/r['class_name']/(r['record_id']+'.png')
            if not p.is_file():
                raise FileNotFoundError(p)
    dim = min(args.pca, len(train)-1, args.size**2)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', dict(
        manifest_sha256=hashlib.sha256(raw).hexdigest(), split_seed=manifest.get('split_seed'),
        seed=args.seed, original_records=dict(counts), classes=classes,
        representation=manifest.get('representation'),
        features=f'grayscale {args.size}x{args.size}, flatten /255; PCA {dim}; StandardScaler',
        preprocessing_fit='train only; validation and test never fit PCA or scaler',
        classifier=dict(kernel='rbf', C=args.c, gamma='scale', probability=False),
        selection='fixed hyperparameters; no validation or test tuning',
        pooling='mean of 12 OVR decision-score vectors, not probabilities' if tiled else 'one image',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={name: __import__(name).__version__ for name in ['numpy', 'sklearn', 'PIL']}))
    summary = []
    for channel in ['uhf', 'hfct']:
        started = time.perf_counter()
        print(f'{channel}: loading {len(train)} clean train images', flush=True)
        x, y = load_features(train, args.data, channel, args.size, classes)
        if set(y) != set(range(4)):
            raise ValueError('Training images must cover four classes')
        model = make_pipeline(PCA(n_components=dim, svd_solver='randomized', random_state=args.seed),
                              StandardScaler(), SVC(kernel='rbf', C=args.c, gamma='scale',
                                                    decision_function_shape='ovr', cache_size=1024))
        print(f'{channel}: fitting PCA and RBF-SVM on train only', flush=True)
        model.fit(x, y)
        del x, y
        joblib.dump(model, args.output/f'svm_{channel}.joblib')
        x, _ = load_features(test, args.data, channel, args.size, classes)
        result = evaluate(model, x, test, classes, tiled)
        if result['n'] != 159:
            raise ValueError('Expected 159 original test records')
        save(args.output/f'test_clean_{channel}.json', result)
        summary.append(dict(method='svm_'+channel, evaluation_split='test', scenario='clean',
                            **{k: result[k] for k in ['n', 'correct', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']}))
        save(args.output/'summary.json', summary)
        print(f"{channel}: records={result['n']} Accuracy={100*result['accuracy']:.2f}% "
              f"Precision(macro)={100*result['precision_macro']:.2f}% "
              f"F1(macro)={100*result['f1_macro']:.2f}% elapsed={time.perf_counter()-started:.1f}s", flush=True)
        del model, x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Existing PRPD directory containing manifest.json and images/clean')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--pca', type=int, default=128)
    parser.add_argument('--c', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=3407)
    main(parser.parse_args())
