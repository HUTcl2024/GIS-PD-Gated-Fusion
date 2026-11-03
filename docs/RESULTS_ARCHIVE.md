# Paper test results — twelve 30° tiles

**Split seed 42; training seed 3407; 159 original test records per condition.** Source: the author's completed server terminal output from `/home/aa/cl-2/runs/tiles_test_seed3407`. Six methods × ten conditions = 60 results. Accuracy (%) below is transcribed to two decimal places, not recomputed from private predictions.

Models were clean-trained and selected using tile-level validation macro-F1. Record predictions average the 12 tile softmax probability vectors. Training source: `runs/tiles_val_seed3407`; data: `revision_output_tiles`. These are not the region/frozen-expert studies preserved below. The same record partition had earlier development use, so this is not an untouched external test set.

| Pollution | SNR dB | UHF model | HFCT model | Concat | Equal | GFM | GFM no confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | — | 98.74 | 98.74 | 98.11 | 100.00 | 98.74 | 98.74 |
| Both | -15 | 35.85 | 23.27 | 28.30 | 25.16 | 25.79 | 25.16 |
| Both | -10 | 23.27 | 23.27 | 14.47 | 25.16 | 24.53 | 25.16 |
| Both | -5 | 25.79 | 0.00 | 5.66 | 25.16 | 25.16 | 12.58 |
| UHF only | -15 | 35.85 | 98.74 | 52.20 | 52.83 | 70.44 | 57.23 |
| UHF only | -10 | 23.27 | 98.74 | 46.54 | 51.57 | 65.41 | 57.23 |
| UHF only | -5 | 25.79 | 98.74 | 46.54 | 63.52 | 65.41 | 51.57 |
| HFCT only | -15 | 98.74 | 23.27 | 48.43 | 48.43 | 48.43 | 48.43 |
| HFCT only | -10 | 98.74 | 23.27 | 48.43 | 48.43 | 45.91 | 48.43 |
| HFCT only | -5 | 98.74 | 0.00 | 48.43 | 47.17 | 47.80 | 48.43 |

The UHF/HFCT columns are separately trained single-channel baseline models, not the internal branch heads of a GFM.

GFM is higher than concat, equal and its no-confidence variant at the three tested UHF-only noise levels. This does not establish superiority across all pollution directions or training seeds. Both-channel results are near chance in these conditions. HFCT-only pollution does not show a consistent GFM advantage. Clean GFM test accuracy is 98.74%; the earlier 100% figure was clean **validation** accuracy.

The packaged test evaluator is `scripts/evaluate_tile_test.py`. Do not overwrite or relabel saved validation results, and do not copy numbers between different partitions or model versions.

## Historical development validation results

The material below is retained as a separate study, not evidence from the paper test run.

<details>
<summary>Expand earlier region and frozen-expert validation results</summary>

# 门控融合结果

**划分 seed 42；训练 seed 3407；158 条原始验证记录。**

以下比较使用同一组冻结的 UHF/HFCT 分支：等权为 logits 等权融合，门控为分类门控（`gate_ce_logits`）。

## 仅 UHF 污染：−5、0、+5 dB

| SNR | 等权 | 门控 | 差值 |
| --- | ---: | ---: | ---: |
| −5 dB | 96.20% | **98.10%** | +1.90 个百分点 |
| 0 dB | 98.10% | 98.10% | 持平 |
| +5 dB | 94.94% | **96.84%** | +1.90 个百分点 |

在这三个已测 SNR 点，门控相较等权取得两次提升、一次持平。

## 双通道与仅 HFCT 污染：−15、−10、−5 dB

| SNR | 双通道：门控领先等权 | 仅 HFCT：门控领先等权 |
| --- | ---: | ---: |
| −15 dB | +6.96 个百分点 | +10.13 个百分点 |
| −10 dB | +5.06 个百分点 | +28.48 个百分点 |
| −5 dB | +12.03 个百分点 | +23.42 个百分点 |

以上展示选定噪声条件下的验证结果。差值按原始正确记录数计算，因此可能与已四舍五入的准确率相减相差 0.01 个百分点。完整 SNR 结果及其他实验放在下方，点击可展开。

<details>
<summary>完整验证结果与实验说明</summary>

**Split seed 42; training seed 3407; 158 original validation records.** These numbers were supplied in the author's server output. Prediction files and checkpoints are not included, so the public snapshot does not independently re-verify those numerical results.

Representation: global plus three 120° views per channel. All four methods share the same frozen clean-trained `region_equal` experts. CE = classification-only gate; KL = reliability gate with additional KL supervision; interval = CE plus the train-only feasible-weight-interval penalty. All entries are accuracy (%).

| Corruption | SNR dB | Equal logits | CE gate | KL gate | Interval gate |
|---|---:|---:|---:|---:|---:|
| Clean | — | 100.00 | 100.00 | 100.00 | 100.00 |
| Both | −15 | 11.39 | 18.35 | 18.35 | 18.35 |
| Both | −10 | 3.80 | 8.86 | 8.86 | 8.86 |
| Both | −5 | 51.27 | 63.29 | 54.43 | 63.29 |
| Both | 0 | 98.10 | 91.14 | 90.51 | 91.14 |
| Both | 5 | 85.44 | 77.22 | 77.22 | 77.22 |
| Both | 10 | 96.20 | 96.20 | 97.47 | 96.20 |
| UHF only | −15 | 95.57 | 94.30 | 93.04 | 94.30 |
| UHF only | −10 | 96.84 | 96.84 | 96.84 | 96.84 |
| UHF only | −5 | 96.20 | 98.10 | 96.84 | 98.10 |
| UHF only | 0 | 98.10 | 98.10 | 98.10 | 98.10 |
| UHF only | 5 | 94.94 | 96.84 | 96.84 | 96.84 |
| UHF only | 10 | 100.00 | 100.00 | 100.00 | 100.00 |
| HFCT only | −15 | 62.66 | 72.78 | 83.54 | 72.78 |
| HFCT only | −10 | 56.33 | 84.81 | 86.08 | 84.81 |
| HFCT only | −5 | 73.42 | 96.84 | 96.84 | 96.84 |
| HFCT only | 0 | 99.37 | 98.73 | 98.73 | 98.73 |
| HFCT only | 5 | 99.37 | 98.73 | 98.73 | 98.73 |
| HFCT only | 10 | 99.37 | 99.37 | 99.37 | 99.37 |

At −5 dB, CE versus equal-logit differences computed from original record counts are +12.03, +1.90 and +23.42 percentage points for both, UHF-only and HFCT-only corruption. Differences computed from rounded displayed accuracies may differ by 0.01 pp.

Favorable discrete conditions were identified after inspecting validation results. They are not proof of superiority throughout a continuous SNR interval. Mean macro-F1 over these same fixed conditions was used to select gate checkpoints. These are not independent final-test results.

The interval and CE gates have identical aggregate accuracies here, which does not imply their parameters, weights or record predictions are identical. No additional accuracy benefit from interval supervision is established. The very low absolute both-channel accuracies at −15 and −10 dB do not establish reliable recognition despite relative gains over equal weighting.

## Separate end-to-end repeated-seed context

A different experiment trained `region_equal` and `region_gfm` end to end on clean data with auxiliary supervision, using seeds 42–46. It does not share the frozen-expert training protocol above. The supplied complete accuracy summaries are retained below to avoid representing seed 3407 as evidence of stable superiority. Mean ± sample SD (%), 158 validation records per seed:

| Corruption | SNR dB | End-to-end equal | End-to-end GFM |
|---|---:|---:|---:|
| Clean | — | 99.87 ± 0.28 | 99.24 ± 0.53 |
| Both | −15 | 16.84 ± 11.44 | 27.09 ± 3.78 |
| Both | −10 | 34.30 ± 9.35 | 29.49 ± 14.31 |
| Both | −5 | 81.90 ± 8.03 | 63.92 ± 13.54 |
| Both | 0 | 79.49 ± 12.79 | 88.35 ± 10.68 |
| Both | 5 | 86.58 ± 10.74 | 85.06 ± 10.72 |
| Both | 10 | 88.35 ± 10.39 | 94.68 ± 1.71 |
| UHF only | −15 | 91.52 ± 10.60 | 82.53 ± 32.72 |
| UHF only | −10 | 96.71 ± 3.89 | 83.54 ± 20.64 |
| UHF only | −5 | 98.48 ± 0.57 | 91.77 ± 10.44 |
| UHF only | 0 | 97.22 ± 2.85 | 93.04 ± 8.55 |
| UHF only | 5 | 97.85 ± 2.03 | 93.92 ± 10.46 |
| UHF only | 10 | 99.75 ± 0.35 | 98.86 ± 0.83 |
| HFCT only | −15 | 47.59 ± 21.27 | 59.87 ± 8.65 |
| HFCT only | −10 | 66.58 ± 15.96 | 65.57 ± 8.06 |
| HFCT only | −5 | 91.14 ± 8.86 | 89.62 ± 4.64 |
| HFCT only | 0 | 91.27 ± 9.51 | 98.99 ± 0.35 |
| HFCT only | 5 | 93.80 ± 11.10 | 97.47 ± 1.73 |
| HFCT only | 10 | 92.41 ± 11.06 | 97.22 ± 2.13 |

Earlier project development inspected the 159-record test partition under another representation. A new random partition of these same records does not make them an untouched independent test dataset.

## Split seed 3407

No completed result has been supplied for the newly requested split. Do not transfer any numbers from the tables above to a split-3407 result table. Refit train-only normalization, regenerate views and noise, and retrain experts and gates first.

</details>


</details>
