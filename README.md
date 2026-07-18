# Vehicle Detection with YOLOv5 — Fine-Tuning, Bug Audit, and Deployment

A YOLOv5 fine-tune on the Vehicles-OpenImages dataset (5 classes: Ambulance, Bus, Car, Motorcycle, Truck), deployed as a live Streamlit app. This repository also documents two data-integrity bugs found and fixed during training, with before/after evidence.

**Live app:** https://vehicle-detection-yolov5.streamlit.app
**Training notebook:** [`YOLOv5_Training_Notebook.ipynb`](./YOLOv5_Training_Notebook.ipynb) — includes full training logs, not just code

---

## Bugs Found and Fixed

Re-running a saved notebook after a long gap surfaced two issues that would have silently invalidated the results if left unchecked.

### 1. Variable mismatch invalidated a training comparison

The notebook trains a layer-frozen YOLOv5m model into a variable `RES_DIR`. Every inference and validation-display call written after it referenced the *previous* run's variable, `RESULTS_DIR`, instead. As a result, every output shown under "frozen layers" was actually the unfrozen model's output — the frozen model's real performance had never been observed.

**Fix:** corrected the three misreferenced calls to use `RES_DIR`. Re-ran inference against the actual frozen-layer weights.

### 2. Duplicate-removal step had no idempotency guard

The dataset-cleaning cell removes every second image (by sorted filename order) to eliminate duplicates:
```python
if (j % 2) == 0:
    os.remove(...)
```
This cell has no check for whether it has already run. Executing it twice in the same session — which happened during a routine "restart and re-run" — silently halved an already-halved dataset, with no error or warning. It was caught by cross-checking the image count reported in a training log (`train: Scanning ... 219 images`) against the expected count for a single, correct deduplication pass (439 images).

**Fix:** restored the dataset from the original archive and re-ran deduplication exactly once, verified against expected counts before any training resumed.

### 3. `data.yaml` path resolution

The Roboflow-exported `data.yaml` used relative paths (`../train/images`) that resolve against YOLOv5's own repository root directory, not the yaml file's location — a source of silent path-resolution failures depending on the exact working directory at runtime. Replaced with absolute paths to remove the ambiguity entirely.

---

## Results — Verified

All numbers below are taken directly from training logs, not estimated. All three runs trained on the identical, correctly-deduplicated dataset (439 train / 125 valid images), 15 epochs, Tesla T4 GPU.

| Metric | Small (YOLOv5s) | Medium Unfrozen (YOLOv5m) | Medium Frozen (first 15 layers) |
|---|---|---|---|
| Parameters | 7.03M | 20.87M | 20.87M |
| GFLOPs | 15.8 | 47.9 | 47.9 |
| Training time | 3.2 min | 4.3 min | 3.4 min |
| mAP50 | 0.545 | 0.635 | 0.612 |
| mAP50-95 | 0.371 | 0.476 | 0.420 |
| Precision | 0.527 | 0.661 | 0.600 |
| Recall | 0.580 | 0.610 | 0.597 |

**Freezing the first 15 layers of YOLOv5m reduced training time by ~21% (4.3 min → 3.4 min) at a cost of 11.8% on mAP50-95 (0.476 → 0.420) and 3.6% on mAP50 (0.635 → 0.612).** A genuine speed/accuracy trade-off — worth it when GPU time is the binding constraint, not worth it when detection accuracy matters more. The deployed app uses the unfrozen medium model (`results_2`/`results_4` in the notebook), the best performer on mAP50 and mAP50-95.

---

## Try It

Upload an image to the [live app](https://vehicle-detection-yolov5.streamlit.app) and get bounding boxes with class labels and confidence scores for any of: Ambulance, Bus, Car, Motorcycle, Truck.

---

## Repository Contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit inference app |
| `best.pt` | Trained YOLOv5m weights (unfrozen run) |
| `requirements.txt` | Python dependencies for deployment |
| `packages.txt` | System-level (apt) dependency required by OpenCV on the deployment host |
| `yolov5_src/` | Bundled YOLOv5 source, used for local model loading without a runtime fetch to GitHub |
| `YOLOv5_Training_Notebook.ipynb` | Full training notebook with logs, including the bugs and fixes documented above |

---

## Running Locally

```bash
git clone https://github.com/pranall/vehicle-detection-yolov5.git
cd vehicle-detection-yolov5
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

[Vehicles-OpenImages](https://public.roboflow.com/object-detection/vehicles-openimages), via Roboflow. 5 classes, 878 training images before deduplication.

## Model

[YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics, fine-tuned from COCO-pretrained weights.