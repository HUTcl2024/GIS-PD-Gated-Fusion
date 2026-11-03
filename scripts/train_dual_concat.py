"""New clean-trained dual-sensor feature-concatenation baselines.

RGB neural branches use the same Paired loader as GFM. SVM uses separate
train-fitted PCA transforms, concatenation and train-fitted scaling. All
predictions are reported at original-record level. No test-based selection.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from revision_train import Paired
from svm_prpd import validate, load_features, metrics, save


class DualConcat(nn.Module):
    def __init__(self, name, classes=4, pretrained=True):
        super().__init__()
        def encoder():
            if name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
                model.fc = nn.Identity()
                return model
            layers = []
            previous = 3
            for width in [32, 64, 128]:
                layers += [nn.Conv2d(previous, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
                previous = width
            return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.uhf, self.hfct = encoder(), encoder()
        dim = 2048 if name == 'resnet50' else 128
        self.head = nn.Sequential(nn.Linear(2*dim, 512), nn.ReLU(),
                                  nn.Dropout(.5), nn.Linear(512, classes))

    def forward(self, u, h):
        return self.head(torch.cat([self.uhf(u), self.hfct(h)], dim=1))


def pooled(scores, records, classes, probabilities):
    groups = {}
    for value, record in zip(scores, records):
        groups.setdefault(record['parent_record_id'], []).append((value, record))
    rows = []
    for parent, children in groups.items():
        if len(children) != 12 or len({r['class_name'] for _, r in children}) != 1:
            raise ValueError('Expected 12 paired blocks with consistent labels')
        score = np.mean([v for v, _ in children], axis=0)
        rows.append(dict(record_id=parent, true=classes.index(children[0][1]['class_name']),
            pred=int(score.argmax()), **{('probabilities' if probabilities else 'decision_scores'): score.tolist()}))
    result = metrics([r['true'] for r in rows], [r['pred'] for r in rows], classes)
    result.update(predictions=rows, metric_unit='original_record',
                  pooling='mean 12 softmax vectors' if probabilities else 'mean 12 OVR SVM score vectors')
    result['tile_level'] = metrics([classes.index(r['class_name']) for r in records],
                                  scores.argmax(axis=1), classes)
    return result


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    output = []
    ids = []
    for u, h, _, batch_ids in loader:
        output.append(model(u.to(device), h.to(device)).softmax(1).cpu().numpy())
        ids.extend(batch_ids)
    if ids != [r['record_id'] for r in loader.dataset.records]:
        raise ValueError('Evaluation data order mismatch')
    return pooled(np.concatenate(output), loader.dataset.records, classes, True)


def print_result(label, result):
    print(f"{label}: records={result['n']} Accuracy={100*result['accuracy']:.2f}% "
          f"Precision(macro)={100*result['precision_macro']:.2f}% "
          f"F1(macro)={100*result['f1_macro']:.2f}%", flush=True)


def svm_run(args, manifest):
    import joblib
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    classes = manifest['classes']
    train = [r for r in manifest['records'] if r['split'] == 'train']
    transforms = []
    parts = []
    for channel in ['uhf', 'hfct']:
        x, y = load_features(train, args.data, channel, 64, classes)
        pca = PCA(n_components=128, svd_solver='randomized', random_state=args.seed)
        parts.append(pca.fit_transform(x))
        transforms.append(pca)
    scaler = StandardScaler()
    features = scaler.fit_transform(np.concatenate(parts, axis=1))
    model = SVC(kernel='rbf', C=10., gamma='scale', decision_function_shape='ovr', cache_size=1024)
    print('SVM: fitting RBF on concatenated train-only PCA features (128+128)', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(features, y)
    del features, parts, x
    joblib.dump(dict(pca=transforms, scaler=scaler, svm=model, classes=classes), args.output/'best.joblib')
    results = {}
    for split, expected in [('val', 158), ('test', 159)]:
        records = [r for r in manifest['records'] if r['split'] == split]
        parts = []
        for channel, pca in zip(['uhf', 'hfct'], transforms):
            x, _ = load_features(records, args.data, channel, 64, classes)
            parts.append(pca.transform(x))
        score = model.decision_function(scaler.transform(np.concatenate(parts, axis=1)))
        result = pooled(score, records, classes, False)
        if result['n'] != expected:
            raise ValueError('Unexpected original-record count')
        save(args.output/f'{split}_clean.json', result)
        print_result('SVM '+split, result)
        results[split] = {k: result[k] for k in ['n', 'accuracy', 'precision_macro', 'f1_macro']}
    save(args.output/'summary.json', dict(method='dual_svm_concat', seed=args.seed,
        split_seed=manifest.get('split_seed'), selection='Fixed hyperparameters; no grid search or validation selection', **results))


def neural_run(args, manifest, device):
    classes = manifest['classes']
    datasets = {s: Paired(manifest, args.data, s) for s in ['train', 'val']}
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=s == 'train', num_workers=0,
        generator=torch.Generator().manual_seed(args.seed)) for s, ds in datasets.items()}
    model = DualConcat(args.method, len(classes), not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    best = -1.
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = count = 0
        for u, h, y, _ in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(u.to(device), h.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(y)
            count += len(y)
        val = evaluate(model, loaders['val'], device, classes)
        if val['n'] != 158:
            raise ValueError('Expected 158 validation records')
        score = val['tile_level']['f1_macro'] if args.selection_unit == 'tile' else val['f1_macro']
        history.append(dict(epoch=epoch, train_loss=loss_sum/count, selection_f1=score,
            **{'val_'+k: val[k] for k in ['n', 'accuracy', 'precision_macro', 'f1_macro']}))
        save(args.output/'history.json', history)
        if score > best:
            best = score
            torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                epoch=epoch, method=args.method, classes=classes), args.output/'best.pt')
            save(args.output/'val_best.json', val)
        scheduler.step()
        print_result(f'{args.method} epoch={epoch} elapsed={time.perf_counter()-start:.1f}s', val)
    checkpoint = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test = Paired(manifest, args.data, 'test')
    result = evaluate(model, DataLoader(test, batch_size=args.batch_size), device, classes)
    if result['n'] != 159:
        raise ValueError('Expected 159 test records')
    save(args.output/'test_clean.json', result)
    print_result(args.method+' TEST', result)
    val = json.loads((args.output/'val_best.json').read_text(encoding='utf-8'))
    save(args.output/'summary.json', dict(method='dual_'+args.method+'_concat', seed=args.seed,
        split_seed=manifest.get('split_seed'), selected_epoch=checkpoint['epoch'],
        **{split: {k: r[k] for k in ['n', 'accuracy', 'precision_macro', 'f1_macro']}
           for split, r in [('val', val), ('test', result)]}))


def main(args):
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; no overwrite')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Positive epochs and batch size required')
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Manifest must match the rendered-data manifest')
    manifest = json.loads(raw)
    tiled, counts = validate(manifest)
    if not tiled or manifest.get('split_seed') != args.expected_split_seed:
        raise ValueError('Expected 12-tile data with the specified split seed')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.method != 'svm' and args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    for split in ['train', 'val']:
        Paired(manifest, args.data, split)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        manifest_sha256=hashlib.sha256(raw).hexdigest(), original_records=dict(counts),
        fusion='feature concatenation; two independent encoders',
        preprocessing='SVM grayscale64 pixels/PCA128 per modality/train-only scaler' if args.method == 'svm' else 'RGB224 ImageNet normalization; no crop/flip/noise',
        initialization='random CNN' if args.method == 'cnn' else 'not applicable SVM' if args.method == 'svm' else 'random' if args.no_pretrained else 'ImageNet',
        selection='fixed C10/gamma scale; no tuning' if args.method == 'svm' else args.selection_unit+' validation macro F1; first maximum',
        optimizer='RBF SVC' if args.method == 'svm' else 'Adam lr0.001 StepLR10 gamma0.1',
        source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
            [('script', Path(__file__)), ('svm_helpers', Path(__file__).with_name('svm_prpd.py')), ('paired_loader', ROOT/'revision_train.py')]}))
    print(f'Dual {args.method}: split_seed={manifest.get("split_seed")}; counts={dict(counts)}', flush=True)
    if args.method == 'svm':
        svm_run(args, manifest)
    else:
        neural_run(args, manifest, torch.device(args.device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-split-seed', type=int, required=True)
    parser.add_argument('--method', choices=['svm', 'cnn', 'resnet50'], required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--selection-unit', choices=['tile', 'record'], default='tile')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    main(parser.parse_args())
