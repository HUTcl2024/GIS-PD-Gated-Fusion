<div align="center">

# Dual-Channel Gated Fusion for GIS Partial Discharge Identification

**Paired UHF and HFCT measurements · Global and regional PRPD views · Waveform-level noise evaluation**

[![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research%20development-blue)](#-release-status)
[![Protocol](https://img.shields.io/badge/new%20split%20seed-3407-green)](#-data-and-protocol)

[Complete historical results](docs/RESULTS.md) · [Reproduction workflow](#-training-and-evaluation)

</div>

## 📰 Abstract

This repository provides research code for partial-discharge defect classification in gas-insulated switchgear (GIS) using paired ultra-high-frequency (UHF) and high-frequency current-transformer (HFCT) measurements. Raw oscilloscope waveforms are converted into phase-resolved partial-discharge (PRPD) representations. Each modality uses a global view and three 120° phase-region views, encoded by independent ResNet50 branches. A lightweight gate combines the modality logits using normalized features and prediction statistics.

The controlled fusion experiments freeze clean-trained modality experts and train only the gate using perturbations of **training waveforms**. Validation compares clean data and fixed AWGN scenarios with both channels, only UHF, or only HFCT corrupted. Equal-logit and equal-probability fusion are separate baselines. The code includes classification, reliability, and exploratory interval-supervised gate objectives.

![Pipeline overview](assets/overview.png)


## 🛠️ Installation

Linux and an NVIDIA GPU are recommended for the full workflow. The recorded server environment used PyTorch **2.9.1+cu128**, torchvision **0.24.1+cu128**, and RTX 4090 GPUs. The Python version on that server has not been independently recorded here; Python 3.11 is suggested for a fresh environment, not represented as the historical version.

```bash
git clone https://github.com/HUTcl2024/GIS-PD-Gated-Fusion.git
cd GIS-PD-Gated-Fusion

conda create -n gis_pd python=3.11 -y
conda activate gis_pd
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python -B -c "import torch, torchvision; import revision_data, revision_train, revision_reliability; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

The CUDA wheel command follows the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/). Supporting package ranges in `requirements.txt` are dependency specifications, not an exact historical environment lock. For an existing experiment, retain its environment and saved protocol.

ResNet initialization can download ImageNet weights. `--no-pretrained` selects random initialization explicitly; it is an experimental change, not a silent fallback. GPU selection uses `CUDA_VISIBLE_DEVICES`; listing two GPUs does not make the scripts distributed training automatically.

## 📂 Data and protocol

Expected **private input** layout:

```text
data/
├── jd/  # protrusion / corona discharge
├── qx/  # void / air-gap discharge
├── xf/  # floating electrode
└── ym/  # particle on insulation / surface discharge
```

WFM captures are paired triplets named `..._ch1.wfm`, `..._ch2.wfm`, and `..._ch3.wfm`. Supported three-channel CSV exports contain TIME, CH1, CH2, and CH3. Channel mapping:

| Channel | Role |
|---|---|
| CH1 | UHF |
| CH2 | Power-frequency phase reference; left uncorrupted |
| CH3 | HFCT |

The WFM reader supports the verified single-frame INT16 `:WFM#003` analog YT format. Other encodings are rejected. The CSV reader validates the export layout and timing. Data are not bundled; reproducing an acquisition-specific result requires the corresponding private data and exact manifest.

The previously audited dataset contains **791 original records**: jd 188, qx 201, xf 200, ym 202. A two-stage stratified 60/20/20 split yields **474 training / 158 validation / 159 test records**. Original captures are the grouping unit; all views and noisy variants inherit their parent's split.

| Quantity | Meaning |
|---|---|
| Split seed | Assigns original records to subsets; the new requested value is **3407** |
| Training seed | Controls initialization and training sampling; principal value **3407** |
| Fixed validation-noise seed | **2026**, hashed with parent identity and channel |
| Validation SNRs | −15, −10, −5, 0, 5, 10 dB |
| Corruption directions | Both, UHF only, HFCT only |
| Prediction unit | Original record; eight region images are not eight independent samples |

The amplitude reference is a per-channel 99th percentile of detected pulse amplitudes from **clean training parents only**. It is fixed for other subsets and noisy data. Changing the split requires refitting these references and regenerating representations. Do not copy old seed-42 references or relabel old results as seed 3407.

For waveform `x`, reference power is `mean((x - median(x))**2)`. Demeaned Gaussian noise is rescaled to `P_noise = P_signal * 10**(-SNR_dB/10)` and added before pulse extraction. Simultaneous corruption uses independent noise per sensor, each at its own requested SNR; it does not require equal absolute noise amplitudes. The reference power includes the captured background, rather than isolated PD-pulse power.

The current partition does not enforce held-out defect specimens or voltage conditions. Repartitioning existing records does not create a newly acquired, untouched external test set.

## 🚀 Training and evaluation

The commands below describe the newer development interface. Server synchronization archives the actual server code without forcing upgrades: if `capabilities.region_resplit`, `explicit_split_seed`, `interval_supervision`, or `parallel_noise_preparation` is false, the corresponding action/flag/objective is unavailable in that snapshot. Do not change or overwrite active experiment scripts merely to satisfy a README example. The optional temporal script is not required for synchronization; any repository copy retained because it is absent on the server is identified separately in provenance.

Run from the repository root. Set paths outside the repository for private data and generated artifacts:

```bash
export RAW_DATA=/path/to/private/data
export WORK=/path/to/private/experiments/split3407
mkdir -p "$WORK"
```

### 1. Prepare an original-record split and training references

For a fresh dataset:

```bash
python -B revision_data.py audit \
  --data "$RAW_DATA" --output "$WORK/parents" \
  --split-seed 3407 --checksum

# This fits references and creates a tile manifest; no tile images are rendered.
python -B revision_data.py tile-fit \
  --data "$RAW_DATA" --parent-manifest "$WORK/parents/manifest.json" \
  --output "$WORK/reference_tiles"

python -B revision_data.py region-fit \
  --parent-manifest "$WORK/reference_tiles/manifest.json" \
  --output "$WORK/regions"
```

The audit rejects invalid captures. If using the existing audited dataset, use its frozen manifest rather than silently excluding additional files. To repartition an existing **region** manifest without rescanning the raw directory:

```bash
# Alternative to the three commands above; use a NEW output directory.
python -B revision_data.py region-resplit \
  --data "$RAW_DATA" \
  --parent-manifest /path/to/existing/regions/manifest.json \
  --output "$WORK/regions" --split-seed 3407
```

`region-resplit` retains record identities, changes membership, and refits references on the new clean training parents. It refuses an existing output directory. The two entry paths need the same record ordering as well as the same seed to reproduce identical memberships; the manifest is authoritative.

### 2. Render clean and fixed noisy validation views

```bash
python -B revision_data.py region-render \
  --data "$RAW_DATA" --output "$WORK/regions" \
  --render-splits train val

bash scripts/render_validation_noise.sh "$RAW_DATA" "$WORK/regions"
```

The noise script renders all **18 validation scenarios**. It does not render test data. Each original record yields four RGB 224×224 views per modality: global 0–360°, 0–120°, 120–240°, and 240–360°. ImageNet normalization is used; phase views are not randomly cropped or flipped.

### 3. Train the clean modality experts

```bash
CUDA_VISIBLE_DEVICES=0 python -B revision_train.py \
  --data "$WORK/regions" --output "$WORK/experts_seed3407" \
  --methods region_equal --seeds 3407 \
  --epochs 30 --batch-size 4 --aux-weight 1.0 \
  --selection-unit record --validation-only
```

The equal-fusion source is trained with fused CE plus the mean branch CE, coefficient 1.0. These experts are jointly trained branches, **not independently trained single-channel baseline models**. The first maximum of clean record-level validation macro-F1 selects the source checkpoint.

### 4. Train gates with the same frozen experts

```bash
CUDA_VISIBLE_DEVICES=0 python -B revision_reliability.py \
  --data "$WORK/regions" --raw-data "$RAW_DATA" \
  --source "$WORK/experts_seed3407" --output "$WORK/gates_seed3407" \
  --gate-modes gate_ce_logits gate_quality_logits gate_interval_logits \
  --seeds 3407 --epochs 30 --batch-size 32 --image-batch 4 \
  --noise-workers 2
```

- Experts and batch-normalization statistics stay frozen.
- Each epoch selects half the training parents for fresh waveform perturbations, with continuous SNR sampled uniformly from [−15, 10] dB.
- Gate minibatches are half clean and half noisy, with approximately balanced noisy directions. Pool sampling is with replacement.
- Two CPU workers can overlap waveform preparation; RAM usage increases. `--noise-workers 0` selects serial execution. This does not enable a second GPU.
- Checkpoint selection uses mean macro-F1 across the same 19 fixed validation conditions. Those results are **development validation**, not independent final-test evidence.
- With three trained gates plus two equal baselines, a complete run saves **95 result rows** (5 methods × 19 scenarios). One trained gate plus two baselines saves 57 rows.
- Gate checkpoints are saved after the training loop. The current gate trainer has **no mid-training resume checkpoint**. Use `tmux` or `nohup` for a long remote run.

Read-only diagnostics on an existing compatible run:

```bash
python -B revision_reliability.py \
  --data "$WORK/regions" --output "$WORK/gates_seed3407" --audit-only

CUDA_VISIBLE_DEVICES=0 python -B revision_reliability.py \
  --data "$WORK/regions" --output "$WORK/gates_seed3407" \
  --diagnose-only --image-batch 4
```

Diagnostics check branch errors, confidence, logit scales and attainable fusion weights. Label-guided attainable-weight values are **oracle diagnostics**, not deployable model accuracies.

## 📊 Results and configurations

### Historical development snapshot — split 42, training seed 3407

The following author-supplied validation results use the older fixed split, 158 original records, the global-plus-three-region representation, and the same frozen clean-trained experts. 

| Corruption at −5 dB | Equal logits | Classification gate | Difference |
|---|---:|---:|---:|
| Both channels | 51.27% | 63.29% | +12.03 pp |
| UHF only | 96.20% | 98.10% | +1.90 pp |
| HFCT only | 73.42% | 96.84% | +23.42 pp |

Both methods obtained 100% clean validation accuracy in that run. Full SNR results, including unfavorable conditions, are in [RESULTS.md](docs/RESULTS.md). For example, under both-channel corruption the classification gate underperformed equal fusion at 0 and 5 dB. The interval loss produced the same aggregate accuracies as the CE gate in all 19 conditions; no added accuracy benefit is claimed. A separate five-seed end-to-end study had mixed outcomes and is not a replication of this frozen-expert experiment.


## 🧩 Modules

| File / symbol | Purpose |
|---|---|
| `revision_data.py` | Capture auditing, record splits, train-only amplitude references, region rendering, waveform AWGN |
| `mso44_wfm.py`, `mso44_csv.py` | WFM and CSV waveform readers |
| `generate_prpd_wfm.py` | Legacy-compatible phase estimation, pulse extraction, amplitude mapping and PRPD plotting |
| `revision_train.py:Paired` | Paired images; region batches have shape `[B,4,3,224,224]` per sensor |
| `revision_train.py:MultiRegionEncoder` | Four independent ResNet50 view encoders; merged 512D modality feature and four logits |
| `revision_reliability.py:ReliabilityFusion` | 1038→64→2 gate, normalized features/probabilities/max probability/margin/entropy |
| `revision_reliability.py:interval_gate_loss` | Exploratory train-only feasible-weight-interval penalty |


Example of the implemented frozen feature interface:

```python
import torch
from revision_reliability import ReliabilityFusion

gate = ReliabilityFusion(dim=512, classes=4, mode='gate_ce_logits')
# Each input concatenates a 512D frozen feature and four modality logits.
u = torch.randn(8, 516)
h = torch.randn(8, 516)
fused_logits, weights = gate(u, h)
assert fused_logits.shape == (8, 4)
assert weights.shape == (8, 2)
```

This demonstrates the interface only; random inputs are not scientific evaluation data.


## 💖 Acknowledgements

The implementation uses [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/stable/), NumPy, SciPy, scikit-learn, Matplotlib and Pillow. 
