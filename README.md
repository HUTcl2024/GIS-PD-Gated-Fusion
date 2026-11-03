# Dual-Channel Gated Fusion for GIS Partial Discharge Identification

Paired UHF/HFCT waveforms → PRPD → twelve paired 30° phase tiles → gated logit fusion → original-record classification.

![Pipeline overview](assets/overview.png)

## Paper experiment

This section documents the **12-tile, clean-trained experiment** used in the current revision. The global-plus-three-region, frozen-expert and noise-trained gate experiments below are separate development studies. Their checkpoints and results are not interchangeable.

| Setting | Paper experiment |
|---|---|
| Original records | 791: jd 188, qx 201, xf 200, ym 202 |
| Train / validation / test | 474 / 158 / 159 original records |
| Split seed / training seed | **42 / 3407** |
| Representation | 12 paired 30° phase tiles per record, 224×224 RGB |
| Training | Clean images, ImageNet initialization, no online noise, no auxiliary branch loss |
| Methods | `uhf`, `hfct`, `concat`, `equal`, `gfm`, `gfm_no_conf` |
| Optimization | Adam, learning rate 0.001; StepLR every 10 epochs, factor 0.1; 30 epochs; batch size 32 |
| Checkpoint selection | Clean validation **tile-level macro-F1**, first maximum |
| Reported evaluation | Mean of 12 tile softmax probability vectors per original record |
| Fixed test noise | AWGN: −15, −10, −5 dB; both / UHF only / HFCT only |

The split is stratified by class at the **original-record level**. Tiles from one record never cross subsets. The implementation does not enforce acquisition-session, physical-specimen or voltage-held-out partitions; independent acquisition is author-declared provenance. The same test partition was inspected under earlier development representations, so it is not an untouched external validation dataset.

### Architecture

Each sensor has a ResNet50 producing a pooled 2048D vector. Two separate paths use this vector:

- Gate feature: `2048 → 512 → ReLU`.
- Classifier: `2048 → 512 → ReLU → Dropout(0.5) → 4 logits`.

Channel confidence is the maximum softmax class probability. Concatenating two 512D gate features and two scalar confidences gives a 1026D input. The gate is `Linear(1026,128) → ReLU → Linear(128,2) → Softmax`. Its two weights sum to one and combine the **branch logits**. Each paired tile is processed separately; only the final class probabilities are averaged across 12 tiles.

`gfm_no_conf` replaces both confidence scalars with zero while retaining the gate dimensions. `equal` jointly trains an average of branch logits. `concat` learns a linear classifier on concatenated 512D features. These end-to-end comparisons do not use the identical frozen experts employed by the separate reliability studies.

### Preprocessing and noise

CH1 = UHF, CH2 = power-frequency phase reference, CH3 = HFCT. WFM triplets and supported three-channel CSV exports are read by `mso44_wfm.py` and `mso44_csv.py`.

`revision_data.py` imports the `legacy_*` helpers from `generate_prpd_wfm.py`: CH2 zero-crossing phase estimation (21-point smoothing), pulse threshold `median(abs(x)) + 8*MAD(abs(x))`, and peak grouping with a 2 μs gap (at least five samples). The tile path does not use the separate high-pass/envelope whole-image routine in that file. The amplitude reference is the pooled 99th percentile of clean **training** pulse amplitudes per channel; the historical split-42 references were UHF 0.019375 V and HFCT 0.04566875 V. Values are transformed by `asinh(2*amplitude/a_ref)/asinh(2)` and clipped to [0,1]. Validation and test never refit the references.

Images use ImageNet normalization, without random cropping or flipping. Noise is added to the **waveform before pulse extraction and drawing**, not to image pixels. Reference power is `mean((x - median(x))**2)` over the complete captured waveform, including existing background. Demeaned Gaussian noise is scaled to the requested power ratio. Both-channel noise is independently generated per channel using each channel's own reference power; CH2 remains clean. The fixed seed is 2026, hashed with record identity and channel. The same channel uses matched noise across the three pollution directions and SNRs.

### Reproduction

Install PyTorch 2.9.1 / torchvision 0.24.1 with the appropriate CUDA wheel, then `pip install -r requirements.txt`. The recorded server used CUDA 12.8 wheels and RTX 4090. Supporting package ranges are not a historical environment lock. Raw data and model files are not bundled.

Use the exact archived original-record manifest to reproduce historical memberships. A random seed alone is insufficient if the record ordering or accepted captures differ. Keep existing manifests, checkpoints and result directories unchanged.

For a fresh tile representation of that archived split:

```bash
python -B revision_data.py tile-fit \
  --data "$RAW_DATA" --parent-manifest "$PARENT_MANIFEST" --output "$TILES"
python -B revision_data.py tile-render --data "$RAW_DATA" --output "$TILES"
```

The parent manifest must be the split-42 manifest. Do not use `region-fit`, `region-resplit` or the split-3407 study for this experiment. Training reproduction (not required to evaluate the saved models):

```bash
CUDA_VISIBLE_DEVICES=1 python -B revision_train.py \
  --data "$TILES" --output "$NEW_TRAIN_RUN" \
  --methods uhf hfct concat equal gfm gfm_no_conf --seeds 3407 \
  --epochs 30 --batch-size 32 --selection-unit tile --validation-only
```

For fixed noisy **test** tiles:

```bash
for snr in -15 -10 -5; do
  for target in both uhf hfct; do
    python -B revision_data.py tile-render \
      --data "$RAW_DATA" --output "$TILES" \
      --noise-split test --snr-db "$snr" --target "$target" || break 2
  done
done
```

Evaluate saved checkpoints without training:

```bash
CUDA_VISIBLE_DEVICES=1 python -B scripts/evaluate_tile_test.py \
  --data "$TILES" --source "$TRAIN_RUN" --output "$NEW_TEST_RUN" \
  --seed 3407 --expected-split-seed 42
```

Historical server paths: `TILES=/home/aa/cl-2/revision_output_tiles`, `TRAIN_RUN=/home/aa/cl-2/runs/tiles_val_seed3407`; the completed terminal evaluation is in `/home/aa/cl-2/runs/tiles_test_seed3407`. **There is no need to rerun that completed evaluation.** The packaged evaluator provides a repeatable entry point using the same `Paired`, `build_model` and `evaluate` functions, with extra preflight checks and provenance. It refuses existing output directories and never updates the source models. A complete run has 60 rows (six models × ten conditions), each with 159 original-record predictions. Its end-to-end execution on the private server data has not yet been repeated as a standalone script.

### Test results and limits

Clean test accuracy: UHF 98.74%, HFCT 98.74%, concat 98.11%, equal 100.00%, GFM 98.74%, GFM without confidence 98.74%.

| UHF-only noise | Concat | Equal | GFM without confidence | GFM |
|---|---:|---:|---:|---:|
| −15 dB | 52.20 | 52.83 | 57.23 | **70.44** |
| −10 dB | 46.54 | 51.57 | 57.23 | **65.41** |
| −5 dB | 46.54 | 63.52 | 51.57 | **65.41** |

Accuracy (%), transcribed from the author's completed server output. [All ten test conditions and six models](docs/RESULTS.md) include unfavorable results. The advantage is conditional on UHF-only pollution in these tested conditions; neither consistent HFCT-only/both-channel gains nor repeated-seed superiority is established. The repository does not include private prediction files or independently re-verify the reported numbers.

### Code map

| File | Role |
|---|---|
| `revision_data.py` | Shared record audit, tile fitting/rendering and waveform AWGN; also contains region exploration |
| `revision_train.py` | `Encoder`, `FusionNet`, `Paired`, tile training and record aggregation |
| `scripts/evaluate_tile_test.py` | Fixed-checkpoint test evaluation, separate output, no training |
| `generate_prpd_wfm.py` | Imported legacy helpers; standalone whole-image entry point is a different preprocessing path |
| `mso44_wfm.py`, `mso44_csv.py` | Raw waveform adapters |
| `recovered_models.py` | Historical concat/equal/Swin baselines, not the paper's six-model table |
| `revision_reliability.py` | Exploratory frozen **region** experts and 1038→64→2 gates; not the paper's 1026→128→2 gate |
| `revision_temporal.py` | Exploratory sliding-window planning/pilot; not part of the paper protocol |
| `scripts/render_validation_noise.sh` | Region validation rendering, not paper tile test rendering |
| `scripts/run_split3407_seed_study.sh` | Separate region/split-3407 study, not required here |
| `scripts/sync_from_server.py` | Allowlisted source synchronization; does not establish result validity |

SVM and a standalone CNN baseline are not implemented in these published experiment scripts. Do not describe their settings as verified by this code.

## Earlier development workflows

The following historical documentation is retained for traceability. It describes **different** representations, training objectives and partitions; it must not be used to interpret the paper test table above.

<details>
<summary>Expand earlier global-plus-three-region / frozen-expert development documentation</summary>

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


</details>
