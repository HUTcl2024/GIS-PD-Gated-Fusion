# Dual-Channel Gated Fusion for GIS Partial Discharge Identification

UHF/HFCT raw waveforms → corrected CH2 phase → complete 360° PRPD images → ResNet50 confidence-gated logit fusion.

![Framework](assets/overview.png)

## Current experiment

| Setting | Value |
|---|---|
| Deduplicated original records | 786 |
| Train / validation / test | 471 / 157 / 158 original records |
| Split / training seed | 42 / 3407; memberships inherited from the saved manifest |
| Input | One complete PRPD per sensor, RGB, 224×224; no phase tiles |
| Amplitude | Raw absolute detected voltage, no asinh/min-max mapping or numerical clipping |
| Vertical axes | Fixed separately per channel: 1.05 × maximum detected clean **training** peak |
| Phase reference | CH2 hysteresis rising-zero crossing; 21-point smoothing; measured periods |
| Training | Clean only, no random crop/flip, no auxiliary loss; ImageNet pixel normalization |
| Fusion models | UHF, HFCT, concatenation, equal logits, GFM, GFM without confidence |
| Selection | Clean record-level validation mean F1; first maximum |
| Noise | Raw-waveform AWGN; SNR −15, −10, −5, 0, 5, 10 dB; UHF-only / HFCT-only / both |

The partition is at original-record level and stratified by class. This code does not impose batch-held-out, voltage-held-out or specimen-held-out partitions. Duplicate captures must be removed **before** creating the supplied original-record manifest. A seed alone is insufficient to reconstruct memberships without the same accepted records and ordering.

## Method

Each ResNet50 supplies a pooled 2048D vector. Separate paths produce a 512D ReLU gate feature and four logits (`2048→512→ReLU→Dropout(0.5)→4`). Channel confidence is the largest softmax probability. The gate concatenates the two 512D features and two confidences: `1026→128→ReLU→2→Softmax`. The final logits are `w_UHF*z_UHF + w_HFCT*z_HFCT`.

`concat` classifies concatenated 512D features; `equal` averages the two branch logits; `gfm_no_conf` sets both confidence inputs to zero while preserving gate dimensions. Parameter names and initialization order are retained for saved checkpoint loading. Single-sensor Swin-T and CNN implementations are retained only for the current baselines.

CH1=UHF, CH2=power-frequency reference, CH3=HFCT. The CH2 hysteresis threshold is 10% of the 99th percentile absolute centered/smoothed reference. Phase zero remains the interpolated rising **zero**, not the hysteresis threshold. Any measured period outside 15–25 ms causes an explicit error; intervals are not silently removed. Incomplete leading/trailing cycles are excluded.

Peak detection remains `median(abs(x))+8*MAD(abs(x))`; above-threshold samples separated by at most 2 μs (minimum five samples) form one group, represented by its largest absolute sample. These are algorithmic detections, not independently verified physical PD counts. No high-pass/envelope filtering is applied.

Raw amplitudes are passed directly to the scatter renderer. Points outside the fixed vertical axis are outside the displayed image and are explicitly counted; they are not compressed or clamped to the image top. RGB pixel standardization for neural networks is distinct from voltage-amplitude normalization.

## Install

Recorded server: Python 3.10, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, RTX4090. Install the CUDA-matched PyTorch pair, then:

```bash
pip install -r requirements.txt
```

Raw waveforms, manifests, trained weights and per-record predictions are private and are not included. The original-record manifest needs `classes`, `split_seed`, `records` and deduplication provenance. Each record needs `record_id`, unique `group_id`, `class_name`, `split`, `source_format`, `sources`, `sample_rate_hz` and `samples`; CSV records also need `source_sha256`. WFM sources are ordered CH1, CH2, CH3. Supported CSV layout is TIME,CH1,CH2,CH3.

## Reproduce in fresh directories

Set paths to the supplied deduplicated **original-record** manifest, waveforms, new image directory and new run directories:

```bash
export RAW_DATA=/path/to/raw_waveforms
export DEDUP_MANIFEST=/path/to/deduplicated_original_records/manifest.json
export DATA=/path/to/new_full360_rawvolts
export FUSION_RUN=/path/to/new_fusion_run
export BASELINE_RUN=/path/to/new_single_sensor_run
export VAL_NOISE_RUN=/path/to/new_val_noise_run
export CLEAN_TEST_RUN=/path/to/new_clean_test_run
export NOISE_TEST_RUN=/path/to/new_noise_test_run

python -u -B scripts/prepare_full360.py   --manifest "$DEDUP_MANIFEST" --raw-data "$RAW_DATA" --output "$DATA"

CUDA_VISIBLE_DEVICES=1 python -u -B revision_train.py   --data "$DATA" --output "$FUSION_RUN"   --methods uhf hfct concat equal gfm gfm_no_conf --seeds 3407   --epochs 30 --batch-size 32 --selection-unit record --aux-weight 0 --validation-only

CUDA_VISIBLE_DEVICES=1 python -u -B scripts/full360_fixed_noise_val.py   --data "$DATA" --raw-data "$RAW_DATA" --source "$FUSION_RUN" --output "$VAL_NOISE_RUN"

CUDA_VISIBLE_DEVICES=1 python -u -B scripts/rawvolts_single_baselines.py   --data "$DATA" --source "$FUSION_RUN" --noise-results "$VAL_NOISE_RUN"   --output "$BASELINE_RUN" --seed 3407 --epochs 30

CUDA_VISIBLE_DEVICES=1 python -u -B scripts/evaluate_rawvolts_clean_test.py   --data "$DATA" --fusion-source "$FUSION_RUN" --baseline-source "$BASELINE_RUN"   --swin-source "$BASELINE_RUN" --output "$CLEAN_TEST_RUN"

CUDA_VISIBLE_DEVICES=1 python -u -B scripts/full360_fixed_noise_test.py   --data "$DATA" --raw-data "$RAW_DATA" --clean-source "$CLEAN_TEST_RUN"   --val-noise-source "$VAL_NOISE_RUN" --output "$NOISE_TEST_RUN"
```

Preparation audits all 786 CH2 references before drawing. Abnormal records are listed and block generation; no record is silently excluded. It fits channel axes on train only, creates 1572 clean full images and records out-of-axis peaks. Training uses train/val only. The baseline script reuses existing independent ResNet50 and dual GFM results and trains SVM/CNN/Swin-T per sensor.

CNN: RGB convolution blocks 32/64/128, GAP and 512D classifier, random initialization. Swin-T: ImageNet initialization, 768→512→ReLU→Dropout(0.5)→4. Deep models use Adam, batch 32, 30 epochs, StepLR(step 10, gamma 0.1). CNN/ResNet learning rate is 0.001; final Swin-T learning rate is 0.00001. SVM uses grayscale 64×64 pixels, train-fitted PCA 128 and StandardScaler, RBF SVC(C=10,gamma=scale); it does not use HOG. Optimization is therefore comparable where applicable, not identical across all model families.

The reported server Swin models came from a separate low-learning-rate run. For those archived checkpoints, pass that directory as `--swin-source`. Fresh reproduction trains Swin directly with the final learning rate in `$BASELINE_RUN`.

## Fixed interference and outputs

SNR is `10*log10(mean((x-median(x))**2)/mean(added_noise**2))`, using the complete captured waveform including existing background, not isolated PD power. Demeaned Gaussian noise is rescaled per record/channel. Seed2026 is hashed with record identity and channel. CH2 is never corrupted. The same standardized noise is used across SNRs for a channel, and identical noisy image bytes are shared between single-channel and both-channel conditions. Unaffected images are copied from clean inputs.

Validation: six fusion-study models × (clean + 18 noise)=114 rows, n=157. Test: twelve frozen models × (clean + 18 noise)=228 rows, n=158. Test evaluation requires the exact clean-test checkpoint hashes, reproduces clean predictions, and checks unaffected single-sensor predictions. White images stay in the evaluation. No optimizer or checkpoint selection is used during interference evaluation.

`comparison.csv`, `main_table.csv` and `ablation_table.csv` report clean comparisons. `all_test_results.csv`, `main_table_by_snr.csv`, `fusion_ablation_by_snr.csv` and `image_quality.csv` report fixed interference results. Accuracy, mean Recall and mean F1 are fractions in CSV files; multiply by 100 for percentages. Mean Recall/F1 are class-equal (macro) averages.

## Code and result provenance

See [reported test results](docs/RESULTS.md) and [source provenance](SOURCE_PROVENANCE.json). This release removes superseded tile/region/temporal/reliability/attention and percentage-noise entry points. It preserves the active model parameter names and numerical PRPD helpers; the current representation uses a single full view.

Use a **fresh checkout and fresh output directories** for this release. Do not overwrite an archived server environment or edit a trained manifest to bypass a hash check. Refactoring changes module byte hashes even when retained numerical functions are identical; old interrupted jobs require their original code snapshot. The private server source files/manifests/checkpoints have not been fetched by this assistant; server executions and scores are author-reported, while local regression checks use synthetic inputs.


## Additional Gaussian-shaped pulse interference

The frozen test extension is `scripts/gaussian_pulse_fixed_test.py`. It keeps the same six SNRs, three corruption directions, 158 test records and twelve frozen checkpoints. It does not add white noise simultaneously. Random pulse centers and independent random signs are fixed per record/channel; each Gaussian-shaped pulse has FWHM1 microsecond, is truncated at +/-4 sigma, and the realized pulse count is round(1000*record duration in seconds). Overlaps are allowed. The pulse train is demeaned, then scaled by its measured full-record mean-square power to the same SNR definition used for AWGN. Width/rate are synthetic evaluation settings, not measured field interference.

Pass `--pulse-fwhm-us 1 --pulse-rate-hz 1000` explicitly. Other required arguments are `--data`, `--raw-data`, `--clean-source` (the completed twelve-model clean-test directory), `--val-noise-source` (the original AWGN validation protocol, for fixed axes/settings) and a fresh `--output`. New images use `test_gaussian_pulse_*`, leaving AWGN scenarios unchanged. Expected output is228 rows including repeated clean controls; no accuracy is reported before server execution.

For archived server experiments, copy **only this self-contained new script** next to the original `revision_train.py`, `revision_data.py` and `generate_prpd_wfm.py` (for example under /home/aa/cl-2). Do not replace those archived modules: their recorded byte hashes remain required. The script also runs from scripts/ in a fresh current checkout.
