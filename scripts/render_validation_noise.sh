#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  printf 'Usage: bash scripts/render_validation_noise.sh RAW_DATA REGION_OUTPUT\n' >&2
  exit 2
fi
raw_data=$1
region_output=$2
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for snr in -15 -10 -5 0 5 10; do
  for target in both uhf hfct; do
    python -B revision_data.py region-render \
      --data "$raw_data" --output "$region_output" \
      --noise-split val --snr-db "$snr" --target "$target"
  done
done
