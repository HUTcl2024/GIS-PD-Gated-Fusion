"""One sensor, one grayscale input channel; explicit manifest; record metrics.

New experiment: not interchangeable with the earlier RGB ResNet50 checkpoints.
Best checkpoint: record-level validation macro F1, first maximum. Test images
are evaluated only after training. No augmentation, no noise, no resplitting.
"""
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def validate(manifest, expected_seed):
    tiled = manifest.get('representation') == 'legacy_phase_tiles_30deg'
    if manifest.get('representation') not in (None, 'whole_record_prpd', 'legacy_phase_tiles_30deg'):
        raise ValueError('Use a whole-record or 12-tile image manifest; region-view manifests are not supported')
    if expected_seed is not None and manifest.get('split_seed') != expected_seed:
        raise ValueError('Manifest split_seed does not match --expected-split-seed')
    records = manifest['records']
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Duplicate image IDs')
    groups = {}
    for r in records:
        key = r['parent_record_id'] if tiled else r['record_id']
        groups.setdefault(key, []).append(r)
    for children in groups.values():
        if len({r['split'] for r in children}) != 1 or len({r['class_name'] for r in children}) != 1:
            raise ValueError('Parent crosses splits/classes')
        if tiled and (len(children) != 12 or
                      {r['phase_start_deg'] for r in children} != set(range(0, 360, 30))):
            raise ValueError('Expected 12 unique phase blocks per parent')
    counts = Counter(v[0]['split'] for v in groups.values())
    if counts != {'train': 474, 'val': 158, 'test': 159}:
        raise ValueError('Expected original-record counts 474/158/159, got ' + str(counts))
    if len(manifest['classes']) != 4:
        raise ValueError('Expected four classes')
    return tiled, dict(counts)


class PRPD(Dataset):
    def __init__(self, manifest, data, split, modality, tiled):
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.classes = manifest['classes']
        self.parents = [r['parent_record_id'] if tiled else r['record_id'] for r in self.records]
        self.paths = [data/'images'/'clean'/modality/r['class_name']/(r['record_id']+'.png')
                      for r in self.records]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing {modality}/{split} images; first: {missing[0]}')
        # Fixed grayscale scaling: no val/test statistics are fitted.
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            x = self.transform(image.convert('L'))
        return x, self.classes.index(self.records[i]['class_name']), self.parents[i]


class CNN(nn.Module):
    """New simple baseline, not a recovered historical CNN."""
    def __init__(self, classes):
        super().__init__()
        layers = []
        previous = 1
        for width in [32, 64, 128]:
            layers += [nn.Conv2d(previous, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                       nn.ReLU(), nn.MaxPool2d(2)]
            previous = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(0.5), nn.Linear(128, classes))

    def forward(self, x):
        return self.net(x)


def build_model(name, classes, pretrained):
    if name == 'cnn':
        return CNN(classes)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    old = model.conv1
    model.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512), nn.ReLU(),
                             nn.Dropout(0.5), nn.Linear(512, classes))
    return model


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    groups = {}
    for x, y, parents in loader:
        probabilities = model(x.to(device)).softmax(dim=1).cpu().numpy()
        for p, label, parent in zip(probabilities, y.tolist(), parents):
            if parent in groups and groups[parent][0] != label:
                raise ValueError('Inconsistent record labels')
            groups.setdefault(parent, (label, []))[1].append(p)
    rows = []
    for parent, (label, blocks) in groups.items():
        p = np.mean(blocks, axis=0)
        rows.append(dict(record_id=parent, true=label, pred=int(p.argmax()),
                         probabilities=p.tolist(), blocks=len(blocks)))
    truth, pred = [r['true'] for r in rows], [r['pred'] for r in rows]
    labels = list(range(len(classes)))
    return dict(n=len(rows), correct=sum(a == b for a, b in zip(truth, pred)),
        accuracy=float(accuracy_score(truth, pred)),
        precision_macro=float(precision_score(truth, pred, labels=labels, average='macro', zero_division=0)),
        f1_macro=float(f1_score(truth, pred, labels=labels, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(truth, pred, labels=labels).tolist(),
        metric_unit='original_record', predictions=rows)


def main(args):
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; no overwrite/resume')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        raise ValueError('Invalid training arguments')
    raw = args.manifest.read_bytes()
    rendered_manifest = args.data/'manifest.json'
    if not rendered_manifest.is_file() or hashlib.sha256(rendered_manifest.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
        raise ValueError('Specified manifest must match the manifest in the rendered-image directory; do not mix old images/references with a new split')
    manifest = json.loads(raw)
    tiled, counts = validate(manifest, args.expected_split_seed)
    print(f'Manifest: {args.manifest.resolve()}; split_seed={manifest.get("split_seed")}; records={counts}', flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu explicitly')
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    datasets = {split: PRPD(manifest, args.data, split, args.modality, tiled) for split in ['train', 'val']}
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == 'train',
        num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == 'cuda') for split, ds in datasets.items()}
    model = build_model(args.model, len(manifest['classes']), not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        manifest_sha256=hashlib.sha256(raw).hexdigest(), split_seed=manifest.get('split_seed'),
        counts=counts, classes=manifest['classes'], input_channels=1,
        preprocessing='grayscale L, resize224, (x/255 - 0.5)/0.5; no random augmentation/no noise',
        initialization='random CNN' if args.model == 'cnn' else 'random' if args.no_pretrained else 'ImageNet; first conv weights averaged over RGB',
        selection='original-record validation macro F1; first maximum wins',
        pooling='arithmetic mean of 12 softmax probability vectors' if tiled else 'one image per record',
        optimizer='Adam', schedule='StepLR step10 gamma0.1',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, torchvision=__import__('torchvision').__version__)
    save(args.output/'protocol.json', protocol)
    best = -1.
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = total = 0
        for x, y, _ in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            total += len(y)
        val = evaluate(model, loaders['val'], device, manifest['classes'])
        if val['n'] != 158:
            raise ValueError('Expected 158 validation records')
        row = dict(epoch=epoch, train_loss=loss_sum/total,
                   **{'val_'+k: val[k] for k in ['n', 'accuracy', 'precision_macro', 'f1_macro']})
        history.append(row)
        save(args.output/'history.json', history)
        if val['f1_macro'] > best:
            best = val['f1_macro']
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(dict(state_dict=state, model=args.model, modality=args.modality, input_channels=1,
                classes=manifest['classes'], best_epoch=epoch, manifest_sha256=protocol['manifest_sha256']), args.output/'best.pt')
            save(args.output/'val_best.json', val)
        scheduler.step()
        print(f"{args.model}/{args.modality} epoch={epoch} val_records=158 "
              f"Acc={100*val['accuracy']:.2f}% Precision={100*val['precision_macro']:.2f}% "
              f"F1={100*val['f1_macro']:.2f}% elapsed={time.perf_counter()-started:.1f}s", flush=True)
    checkpoint = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test = PRPD(manifest, args.data, 'test', args.modality, tiled)
    result = evaluate(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.workers),
                      device, manifest['classes'])
    if result['n'] != 159:
        raise ValueError('Expected 159 test records')
    save(args.output/'test_clean.json', result)
    summary = dict(model=args.model, modality=args.modality, seed=args.seed,
        split_seed=manifest.get('split_seed'), selected_epoch=checkpoint['best_epoch'],
        evaluation_split='test', **{k: result[k] for k in ['n', 'correct', 'accuracy', 'precision_macro', 'f1_macro']})
    save(args.output/'summary.json', summary)
    print(f"TEST {args.model}/{args.modality}: n=159 Acc={100*result['accuracy']:.2f}% "
          f"Precision(macro)={100*result['precision_macro']:.2f}% F1(macro)={100*result['f1_macro']:.2f}%", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='PRPD root containing images/clean/uhf and images/clean/hfct')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-split-seed', type=int, required=True)
    parser.add_argument('--modality', choices=['uhf', 'hfct'], required=True)
    parser.add_argument('--model', choices=['resnet50', 'cnn'], default='resnet50')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--no-pretrained', action='store_true')
    main(parser.parse_args())
