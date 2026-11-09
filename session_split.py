"""Explicit session-grouped splitting and class-specific unseen-voltage checks."""
import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def voltage(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid voltage_kv') from exc
    if not result.is_finite() or result <= 0:
        raise ValueError('voltage_kv must be finite and positive')
    return str(result.normalize())


def holdouts(config, classes):
    if set(config) != set(classes):
        raise ValueError('Provide holdout voltages for every class, and no other classes')
    result = {}
    for label, values in config.items():
        if not isinstance(values, list) or not values:
            raise ValueError('Each class requires a nonempty voltage list')
        result[label] = {voltage(v) for v in values}
    return result


def validate_grouped_manifest(manifest):
    """Reject record leakage, session leakage, and overlap of held-out voltages."""
    protocol = manifest.get('split_protocol', {})
    if not isinstance(protocol, dict) or protocol.get('name') != 'session_grouped_unseen_voltage_v1':
        raise ValueError('Create a session-grouped unseen-voltage manifest first')
    classes = manifest['classes']
    held = holdouts(protocol['holdout_voltages_kv'], classes)
    records = manifest['records']
    counts = Counter()
    sessions = {}
    ids = set()
    coverage = defaultdict(set)
    trainval_voltages = defaultdict(set)
    test_voltages = defaultdict(set)
    for record in records:
        rid = record['record_id']
        if not isinstance(rid, str) or not rid or rid in ids or 'parent_record_id' in record:
            raise ValueError('Expected unique original record IDs')
        ids.add(rid)
        sid = record.get('session_id')
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError('Every record needs a verified session_id')
        split, label = record['split'], record['class_name']
        if split not in ('train', 'val', 'test') or label not in classes:
            raise ValueError('Unknown split or class')
        if sid in sessions and sessions[sid] != split:
            raise ValueError('Acquisition session crosses subsets: ' + sid)
        sessions[sid] = split
        v = voltage(record['voltage_kv'])
        if v in held[label] and split != 'test':
            raise ValueError('Held-out voltage leaked into development data')
        (test_voltages if split == 'test' else trainval_voltages)[label].add(v)
        counts[split] += 1
        coverage[split].add(label)
    for label in classes:
        if held[label] - test_voltages[label]:
            raise ValueError('Configured holdout voltage is absent: ' + label)
        if test_voltages[label] & trainval_voltages[label]:
            raise ValueError('Test voltage also occurs in train/val: ' + label)
    for split in ('train', 'val', 'test'):
        if coverage[split] != set(classes):
            raise ValueError('Every subset must contain all defect classes: ' + split)
    return dict(counts)


def make_split(manifest, metadata, config):
    result = copy.deepcopy(manifest)
    records = result['records']
    ids = [r['record_id'] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(metadata):
        raise ValueError('Metadata must match the unique manifest record IDs exactly')
    classes = result['classes']
    held = holdouts(config['holdout_voltages_kv'], classes)
    validation = config['validation_session_ids']
    if not isinstance(validation, list) or not validation or len(set(validation)) != len(validation):
        raise ValueError('Provide unique validation_session_ids from acquisition logs')
    validation = set(validation)
    sessions = defaultdict(list)
    for record in records:
        row = metadata[record['record_id']]
        sid = row['session_id'].strip()
        if not sid:
            raise ValueError('Empty session_id')
        v = voltage(row['voltage_kv'])
        if record.get('voltage_kv') is not None and voltage(record['voltage_kv']) != v:
            raise ValueError('Voltage conflicts with original manifest: ' + record['record_id'])
        record.update(session_id=sid, voltage_kv=v)
        sessions[sid].append(record)
    if validation - sessions.keys():
        raise ValueError('Unknown validation session IDs')
    testing = {sid for sid, group in sessions.items()
               if any(r['voltage_kv'] in held[r['class_name']] for r in group)}
    if validation & testing:
        raise ValueError('A validation session contains a held-out voltage')
    for sid, group in sessions.items():
        split = 'test' if sid in testing else 'val' if sid in validation else 'train'
        for record in group:
            record['split'] = split
    result.pop('split_seed', None)
    result['split_protocol'] = dict(
        name='session_grouped_unseen_voltage_v1',
        holdout_voltages_kv={k: sorted(v) for k, v in held.items()},
        validation_session_ids=sorted(validation),
        evaluation_unit='original_record',
        session_policy='Entire session goes to test if it contains a held-out voltage',
        voltage_policy='All test voltages must be absent from train and validation within each class')
    counts = validate_grouped_manifest(result)
    result['split_summary'] = {
        split: dict(records=counts[split],
                    sessions=sorted({r['session_id'] for r in records if r['split'] == split}),
                    classes={label: dict(
                        records=sum(r['split'] == split and r['class_name'] == label for r in records),
                        voltages_kv=sorted({r['voltage_kv'] for r in records
                                            if r['split'] == split and r['class_name'] == label}))
                             for label in classes})
        for split in ('train', 'val', 'test')}
    return result


def main(args):
    raw = args.manifest.read_bytes()
    meta_raw = args.metadata.read_bytes()
    config_raw = args.config.read_bytes()
    with args.metadata.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    metadata = {r['record_id']: r for r in rows}
    if len(metadata) != len(rows):
        raise ValueError('Duplicate metadata record IDs')
    result = make_split(json.loads(raw), metadata, json.loads(config_raw))
    result['split_protocol']['source_sha256'] = {
        'manifest': hashlib.sha256(raw).hexdigest(),
        'metadata': hashlib.sha256(meta_raw).hexdigest(),
        'config': hashlib.sha256(config_raw).hexdigest()}
    # New manifest only. Existing splits, images, and results are never overwritten.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(validate_grouped_manifest(result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
