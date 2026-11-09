# Test results — 224×224 scatter PRPD

The experiment uses 786 original records with a fixed train/validation/test split of 471/157/158 (approximately 6:2:2), split seed 42 and training seed 3407. Inputs are paired UHF/HFCT complete 360° PRPD scatter images resized to 224×224, with corrected CH2 phase and fixed per-channel amplitude axes fitted on the training set.

The four tables below transcribe the author-provided manuscript tables. All metrics are percentages. Macro Recall and Macro F1-score are class-equal averages. These are single-training-seed results.

## Table 2. Single-channel baselines on the clean test set

| Method | Input modality | Accuracy (%) | Macro Recall (%) | Macro F1-score (%) |
|---|---|---:|---:|---:|
| SVM | UHF | 89.87 | 90.15 | 90.03 |
| CNN | UHF | 96.84 | 96.92 | 96.82 |
| ResNet50 | UHF | 96.84 | 96.92 | 96.83 |
| Swin-T | UHF | 97.47 | 97.53 | 97.47 |
| SVM | HFCT | 79.11 | 79.44 | 76.54 |
| CNN | HFCT | 96.20 | 96.31 | 96.30 |
| ResNet50 | HFCT | 97.47 |97.53 | 97.47 |
| Swin-T | HFCT | 96.84 | 96.94 | 96.91 |

## Table 4. Classification accuracy under AWGN (%)

| Corrupted channel(s) | SNR (dB) | Feature concatenation | Equal-weight fusion | GFM | GFM without confidence inputs |
|---|---:|---:|---:|---:|---:|
| UHF only | -5 | 91.14 | 56.33 | 97.47 | 96.20 |
| UHF only | 0 | 91.14 | 68.35 | 96.84 | 96.20 |
| UHF only | +5 | 94.30 | 95.57 | 96.84 | 96.20 |
| HFCT only | -5 | 89.87 | 97.47 | 97.47 | 71.52 |
| HFCT only | 0 | 96.84 | 97.47 | 97.47 | 90.51 |
| HFCT only | +5 | 96.84 | 97.47 | 96.84 | 93.67 |
| Both channels | -5 | 91.14 | 50.63 | 97.47 | 71.52 |
| Both channels | 0 | 97.47 | 68.99 | 96.84 | 90.51 |
| Both channels | +5 | 98.10 | 96.84 | 96.20 | 93.67 |

## Table 5. Dual-channel fusion ablation on the clean test set

| Fusion method | Input modality | Accuracy (%) | Macro Recall (%) | Macro F1-score (%) |
|---|---|---:|---:|---:|
| Feature concatenation | UHF + HFCT | 97.47 | 97.53 | 97.48 |
| Equal-weight fusion | UHF + HFCT | 96.84 | 96.92 | 96.91 |
| GFM (Ours) | UHF + HFCT | 97.47 | 97.53 | 97.53 |
| GFM without confidence inputs | UHF + HFCT | 96.20 | 96.31 | 96.25 |

## Table 6. Classification accuracy under Gaussian pulse interference (%)

| Corrupted channel(s) | SNR (dB) | Feature concatenation | Equal-weight fusion | GFM | GFM without confidence inputs |
|---|---:|---:|---:|---:|---:|
| UHF only | -5 | 91.77 | 96.20 | 97.47 | 96.20 |
| UHF only | 0 | 94.30 | 77.22 | 97.47 | 96.20 |
| UHF only | +5 | 98.10 | 95.57 | 97.47 | 96.20 |
| HFCT only | -5 | 96.84 | 94.94 | 94.30 | 80.38 |
| HFCT only | 0 | 91.77 | 96.84 | 91.77 | 81.01 |
| HFCT only | +5 | 90.51 | 97.47 | 97.47 | 74.05 |
| Both channels | -5 | 89.87 | 92.41 | 94.30 | 75.32 |
| Both channels | 0 | 85.44 | 77.22 | 91.77 | 80.38 |
| Both channels | +5 | 91.14 | 95.57 | 94.94 | 74.05 |
