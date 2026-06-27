# Facial Emotion Recognition with ViT

**🔴 [Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/barisgovercin/facial-emotion-vit)** — upload a face photo and the model predicts the emotion.

Fine-tuning a Vision Transformer (ViT-Base/16) to classify emotions from face crops extracted from RAVDESS videos. Five emotions: angry, fearful, happy, neutral, sad.

This was built as part of CE903 Group Project at the University of Essex. The model runs frame-level classification and can aggregate predictions at video level using majority voting or mean probability.

## Results

| Eval Condition | Accuracy | Notes |
|---|---|---|
| Test set (10,796 frames) | 74.2% | 3 unseen actors, TTA enabled |
| Manual test (4,887 frames) | 85.9% | Actor_03, Actor_10, Actor_18 |
| Video-level (216 videos) | 94.9% | Mean probability aggregation |
| Top-2 accuracy | 89.6% | Correct label in top 2 predictions |
| Top-3 accuracy | 95.6% | Correct label in top 3 predictions |

Happy is the easiest class (F1=0.95), neutral is the hardest (F1=0.55). Most errors happen between visually similar emotions like fearful/angry and neutral/sad.

## Project Structure

```
EMOTION/
├── data/
│   ├── ravdess_raw/              # raw RAVDESS actor folders (Actor_01..Actor_24)
│   ├── raw_videos/               # sorted by emotion label after prepare step
│   ├── processed_faces/          # 224x224 face crops for training
│   ├── train.json                # training split manifest
│   ├── val.json                  # validation split manifest
│   ├── test.json                 # test split manifest
│   └── manual_test/
│       ├── ravdess_raw/          # raw videos from hold-out actors
│       ├── processed_faces/      # face crops from hold-out actors
│       ├── manual_test.json      # manifest for manual test
│       ├── crema/                # CREMA-D videos (cross-dataset test)
│       ├── crema_processed_faces/
│       └── crema_test.json
├── models/
│   └── vit_baseline.py           # ViT-Base/16 model definition
├── scripts/
│   ├── prepare_ravdess.py        # step 1: sort videos into label folders
│   ├── extract_faces.py          # step 2: haar cascade face extraction
│   ├── make_split.py             # step 3: actor-disjoint train/val/test split
│   ├── train_vit.py              # step 4: two-phase training
│   ├── eval_vit.py               # step 5: frame-level evaluation with TTA
│   ├── eval_video_level.py       # step 6: video-level aggregation
│   ├── plot_results.py           # step 7: generate all plots
│   ├── infer.py                  # webcam / single image inference
│   ├── extract_manual_faces.py   # face extraction for manual test actors
│   ├── make_manual_test_json.py  # build manual test manifest
│   ├── crema_extract_faces.py    # face extraction for CREMA-D
│   └── make_crema_test_json.py   # build CREMA-D test manifest
├── dataset_vit.py                # PyTorch Dataset class (JSON-based)
├── outputs/
│   ├── vit_best.pth              # best checkpoint (epoch 11, val acc 71%)
│   ├── vit_last.pth              # latest checkpoint
│   ├── training_log.json         # per-epoch metrics
│   ├── test_report.json          # test set evaluation results
│   ├── manual_test_report.json   # manual test evaluation results
│   ├── video_level_report.json   # video-level aggregation results
│   ├── test_predictions.json     # per-frame prediction log
│   └── plots/                    # all generated figures
└── README.md
```

## Setup

Tested on Python 3.11 with CUDA. Should also work on CPU but training will be slow.

```bash
pip install torch torchvision timm opencv-python pillow matplotlib scikit-learn tqdm
```

### Data Placement

Before running anything, you need to put the RAVDESS data in the right folders. The dataset comes as `Actor_XX` folders (speech videos) and `Video_Song_Actor_XX` folders (song videos).

**For training/evaluation** — put all actor folders into `data/ravdess_raw/`:

```
data/
└── ravdess_raw/
    ├── Actor_01/
    ├── Actor_02/
    ├── ...
    ├── Actor_24/
    ├── Video_Song_Actor_01/
    ├── Video_Song_Actor_02/
    ├── ...
    └── Video_Song_Actor_24/
```

**For manual testing** — put the hold-out actors into `data/manual_test/ravdess_raw/`. These are the actors you want to test on separately (they should NOT be in the training data). We used Actor_03, Actor_10, and Actor_18:

```
data/
└── manual_test/
    └── ravdess_raw/
        ├── Actor_03/
        ├── Actor_10/
        ├── Actor_18/
        ├── Video_Song_Actor_03/
        ├── Video_Song_Actor_10/
        └── Video_Song_Actor_18/
```

The `prepare_ravdess.py` script reads from `data/ravdess_raw/` and the `extract_manual_faces.py` script reads from `data/manual_test/ravdess_raw/`. If these folders are empty or missing, the scripts will throw an error.

## How to Run

The pipeline has a specific order. Each step depends on the previous one.

### 1. Organize videos into emotion folders

```bash
python scripts/prepare_ravdess.py
```

Reads filenames like `01-01-05-02-01-01-12.mp4`, extracts the emotion code, and copies each video to `data/raw_videos/<label>/`. Prefixes filenames with actor ID to avoid collisions.

### 2. Extract face crops

```bash
python scripts/extract_faces.py
```

Runs Haar Cascade face detection on every 5th frame, keeps the largest face, resizes to 224x224, saves as JPEG. Produces about 78K face crops total.

### 3. Create train/val/test splits

```bash
python scripts/make_split.py
```

Actor-disjoint split with seed=42. No actor appears in more than one split. Ratio is 70/20/10 (roughly 54K/14K/10K frames). Writes `data/train.json`, `data/val.json`, `data/test.json`.

### 4. Train the model

```bash
python scripts/train_vit.py
```

Two-phase training:
- Phase 1 (epochs 1-3): backbone frozen, only the classification head trains. Learning rate warms up from 10% to 100%.
- Phase 2 (epochs 4-30): everything unfreezes, MixUp and CutMix kick in. Cosine annealing drops the learning rate toward 1e-7.

Focal Loss with gamma=2.0 and WeightedRandomSampler handle the class imbalance (neutral has half the samples of other classes). Early stopping with patience=8. Training typically stops around epoch 20, with the best checkpoint at epoch 11.

Saves `outputs/vit_best.pth` (best val accuracy) and `outputs/vit_last.pth` (latest state).

### 5. Evaluate on test set

```bash
python scripts/eval_vit.py
```

Runs the test split through the model with 4-view TTA (original, flipped, two corner crops). Reports accuracy, top-2/3 accuracy, confusion matrix, per-class precision/recall/F1, and confidence statistics. Also saves a per-frame prediction log for error analysis.

### 6. Evaluate at video level

```bash
python scripts/eval_video_level.py --json data/manual_test/manual_test.json --ckpt outputs/vit_best.pth
```

Groups frame predictions by source video and aggregates using majority voting and mean probability. Mean probability gave 205/216 correct (94.9%), majority voting gave 199/216 (92.1%).

### 7. Generate plots

```bash
python scripts/plot_results.py
```

Reads the JSON reports and produces training curves, confusion matrices, per-class metric charts, and video-level accuracy bar chart. Saves everything under `outputs/plots/`.

## Manual Test Pipeline

The manual test uses actors that never appeared in training, validation, or test splits.

```bash
# extract faces from hold-out actors
python scripts/extract_manual_faces.py

# build the json manifest
python scripts/make_manual_test_json.py

# run frame-level evaluation
python scripts/eval_vit.py --json data/manual_test/manual_test.json --ckpt outputs/vit_best.pth

# run video-level evaluation
python scripts/eval_video_level.py --json data/manual_test/manual_test.json --ckpt outputs/vit_best.pth
```

## CREMA-D Cross-Dataset Test

To test generalization on a completely different dataset:

```bash
# put CREMA-D videos under data/manual_test/crema/
python scripts/crema_extract_faces.py
python scripts/make_crema_test_json.py
python scripts/eval_vit.py --json data/manual_test/crema_test.json --ckpt outputs/vit_best.pth
```

Accuracy drops to around 50% on CREMA-D, which is expected given the domain gap (different actors, recording setup, lighting conditions).

## Live Inference

```bash
# webcam
python scripts/infer.py --webcam

# single image
python scripts/infer.py --image path/to/face.jpg

# iphone camera stream
python scripts/infer.py --webcam --iphone http://10.130.234.49:8080
```

The webcam mode runs face detection and prediction in a background thread so the display loop stays smooth. Shows a probability bar chart overlay and the predicted emotion label above the bounding box.

## Key Design Decisions

**ViT over CNN:** Self-attention gives a global receptive field from the first layer, which helps when discriminative features (eyebrow position, mouth shape, eye openness) are spread across the face. CNNs need many stacked layers to propagate information between distant spatial regions.

**Two-phase training:** Unfreezing the backbone from epoch 1 destroyed the pretrained features because the random head produced large, noisy gradients. Freezing for 3 epochs lets the head stabilize first.

**Focal Loss + WeightedRandomSampler:** These tackle class imbalance from different angles. The sampler controls how often each class appears per epoch (frequency). Focal Loss controls how much each misclassification contributes to the gradient (magnitude). Using both together works better than either alone.

**MixUp + CutMix:** Applied only after backbone unfreezes. MixUp smooths global decision boundaries between similar emotions. CutMix forces the model to classify from partial face information. Together they reduced the train-val accuracy gap from 32% to about 12%.

**TTA (4 views):** Averaging predictions across original, flipped, and two corner crops improves accuracy by about 1.5% at the cost of 4x inference time. Worth it for evaluation, not used in real-time webcam mode.

## Known Limitations

- Neutral is consistently the weakest class because it overlaps with the start/end frames of every emotion video where the actor's expression fades.
- The model is trained on RAVDESS only (24 actors in a controlled studio). It does not generalize well to other datasets or in-the-wild conditions.
- Frame-level classification misses temporal dynamics. A fearful expression that builds over several frames looks neutral in early frames.
- Haar Cascade occasionally misses faces or crops them off-center, which adds noise to both training and evaluation.
