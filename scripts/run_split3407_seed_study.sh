#!/usr/bin/env bash
# Development validation study, not independent final-test evidence.
# Complete the fixed training-seed set 42..46; retain existing seed 3407 separately.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_split3407_seed_study.sh GPU TRAIN_SEED [TRAIN_SEED ...]' >&2
  exit 2
fi
gpu=$1
shift
export CUDA_VISIBLE_DEVICES="$gpu"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
data=/home/aa/cl-2/revision_output_regions_split3407
raw=/home/aa/cl-2/first_paper_data/data
study=/home/aa/cl-2/runs/seed_study_split3407
declare -A seen=()
for seed in "$@"; do
  case "$seed" in 42|43|44|45|46) ;; *) echo "Unexpected seed: $seed" >&2; exit 2;; esac
  if [[ -n ${seen[$seed]+yes} ]]; then echo "Duplicate seed: $seed" >&2; exit 2; fi
  seen[$seed]=1
  if [[ -e "$study/seed$seed" ]]; then
    echo "STOP: existing seed directory; no overwrite or implicit resume: $study/seed$seed" >&2
    exit 1
  fi
done

# Validate all inputs before creating any experimental output. Never load test images.
python -B - "$data" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
import torch
from revision_train import Paired
from revision_reliability import SCENARIOS, GATE_MODES
p=Path(sys.argv[1]); m=json.loads((p/'manifest.json').read_text())
assert m['split_seed']==3407
assert m['representation']=='legacy_global_three_regions'
assert Counter(r['split'] for r in m['records'])=={'train':474,'val':158,'test':159}
assert 'feature_concat' in GATE_MODES
assert torch.cuda.is_available(), 'Selected CUDA GPU unavailable'
Paired(m,p,'train')
for scenario in SCENARIOS:
    Paired(m,p,'val',scenario)
    if scenario!='clean':
        config=json.loads((p/'images'/scenario/'config.json').read_text())
        assert config['split']=='val'
print('Preflight passed: split3407, clean train and all 19 validation conditions; no test images.',flush=True)
print('Selected GPU:',torch.cuda.get_device_name(0),flush=True)
PY

for seed in "$@"; do
  folder="$study/seed$seed"
  mkdir -p -- "$study"
  mkdir -- "$folder"
  echo "Training clean experts: training seed=$seed, split seed=3407"
  python -B revision_train.py \
    --data "$data" --output "$folder/experts" \
    --methods region_equal --seeds "$seed" --epochs 30 --batch-size 4 \
    --aux-weight 1.0 --selection-unit record --validation-only

  for mode in gate_ce_logits feature_concat; do
    echo "Training frozen-expert fusion: $mode, seed=$seed"
    python -B revision_reliability.py \
      --data "$data" --raw-data "$raw" --source "$folder/experts" \
      --output "$folder/$mode" --gate-modes "$mode" --seeds "$seed" \
      --epochs 30 --batch-size 32 --image-batch 4 --noise-workers 2
  done

  python -B - "$folder" "$seed" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); seed=int(sys.argv[2]); lookups={}; origins=[]
for mode in ['gate_ce_logits','feature_concat']:
    rows=json.loads((root/mode/'results.json').read_text())
    assert len(rows)==57
    assert len({(r['method'],r['seed'],r['scenario']) for r in rows})==57
    assert all(r['seed']==seed and r['n']==158 for r in rows)
    lookups[mode]={(r['method'],r['scenario']):r for r in rows}
    detail=json.loads((root/mode/f'{mode}_seed{seed}'/'val_clean.json').read_text())
    origins.append(detail['frozen_source'])
assert origins[0]==origins[1], 'Different frozen experts'
gate,concat=lookups['gate_ce_logits'],lookups['feature_concat']
scenarios=[s for method,s in gate if method=='gate_ce_logits']
for scenario in scenarios:
    for baseline in ['equal_logits','equal_probs']:
        assert gate[baseline,scenario]['correct']==concat[baseline,scenario]['correct']
print(f'COMPLETED seed={seed}: experts and two fusion heads; baseline counts and source match.',flush=True)
for mode,lookup in lookups.items():
    selected=[r for (method,s),r in lookup.items() if method==mode]
    score=sum(r['macro_f1'] for r in selected)/len(selected)
    print(f'{mode}: mean macro F1 across 19 selection-validation conditions={100*score:.2f}%',flush=True)
print('Retain every seed. Any best-seed illustration is post-hoc; do not substitute it for repeated-seed results.',flush=True)
PY
done
echo 'Assigned seed jobs finished. Existing seed3407 results remain unchanged.'
