# Recovered baseline source audit

The author supplied SHA256 hashes from `/home/aa/resnet/数据处理/`. All five files below match the inspected local source bytes. This establishes source identity, not training completion or model accuracy.

| Original file | Role | SHA256 |
|---|---|---|
| `swin_tyzs.py` | Single-channel Swin-T | `06569b1ee0f9d70bbd9c9f4034ccb934ced2e05591ff8e0fa81427e82f708238` |
| `train.py` | Single-channel ResNet50 | `fa22494bd8347a9b18318766b5d7355779585d9a20045ceb19406b49c9ec3253` |
| `model.py` | ResNet50 with a different training schedule, not a separate shallow CNN | `c4bd1e8cfa011bd8f7718c1329074429e1fb266a89ef3d747e2285686b028bc1` |
| `fusion_concat.py` | Dual-channel feature concatenation | `314b89c7894e1e9c2a8f94a62916182d3304998f6a86d77d4cbd6ad554dbdb6c` |
| `dqrh.py` | Dual-channel equal-logit fusion | `9e0348dfe476575e905e071f588a100bf9f8299c3ba2781b4c6203af86b4a802` |

## Architectures retained

- `swin_uhf` / `swin_hfct`: torchvision Swin-T with a `768 → 512 → ReLU → Dropout(0.5) → 4` head. The wrapper selects the requested modality. Definition: `RecoveredSwin` in `recovered_models.py`.
- `recovered_concat`: two ResNet50 pooled 2048D vectors, independent `2048 → 512 → ReLU` projections, then `Dropout(0.5) → Linear(1024,512) → ReLU → Dropout(0.5) → Linear(512,4)`.
- `recovered_equal`: two ResNet50 pooled 2048D vectors, each with `Dropout(0.5) → Linear(2048,4)`; average their logits.
- `uhf` / `hfct`: the current paper's ResNet50 classifier head matches the `2048 → 512 → ReLU → Dropout(0.5) → 4` classification path in `train.py`. The common wrapper also instantiates a gate feature projection; it is unused by single-channel forward inference.

`concat` and `equal` in the six-method paper experiment have different heads from `recovered_concat` and `recovered_equal`. Do not interchange names or results.

## Adaptations for controlled comparisons

Only architecture definitions are recovered. Use `revision_train.py` for the archived record split, paired phase tiles, train-only amplitude references, RGB 224×224 inputs, ImageNet normalization, optimizer, training budget and checkpoint selection. Test predictions average 12 tile probability vectors per original record.

The old Swin script used random image cropping/flipping and pixel Gaussian noise during training; these are not retained in the clean-trained paper protocol. Its fallback to random initialization after a pretrained-weight download failure is also not retained: the common implementation raises an error rather than silently changing initialization. `--no-pretrained` is an explicit experimental alternative.

The original `model.py` uses staged freezing/fine-tuning. That schedule is not represented as identical to the unified full-model optimization. Reusing a classifier structure does not reproduce its historical training schedule or historical accuracy.

## Entry points and status

`revision_train.py --methods` already supports all four optional recovered baselines. For the paper tile protocol, explicitly use training seed 3407 and `--selection-unit tile --validation-only`, with the archived split-42 tile manifest. Optional baselines require their own trained checkpoints; weights from other splits, whole-image inputs or region studies cannot substitute.

`scripts/evaluate_tile_test.py` now accepts these methods via `--methods` as well. Its default remains the original six paper methods. It refuses missing checkpoints and mismatched manifests. It does not train a missing baseline.

The source inventory and filename search supplied by the author did not locate a distinct shallow-CNN or SVM implementation. ResNet50 is a convolutional network, but it should not be relabeled as an additional independent CNN baseline. No new CNN/SVM architecture, training run or result is claimed by this update. No optional split-42 / training-3407 tile baseline result has been verified in this update.
