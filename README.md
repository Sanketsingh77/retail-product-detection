# Retail Product Detection

End-to-end multi-class grocery product detection using PyTorch/YOLO,
with dataset validation, controlled model comparison, ONNX export,
CPU benchmarking, and FastAPI inference.

## Problem

Retail computer-vision systems need to identify and localize products
under variations in viewpoint, occlusion, truncation, and packaging.

This project builds a 25-class grocery product detector and follows the
complete pipeline:

Dataset → Validation → Training → Evaluation → Error Analysis →
Model Comparison → ONNX Export → Benchmarking → FastAPI

## Example Predictions

The final YOLO11s model detects and classifies grocery products with
bounding boxes and confidence scores.

<p align="center">
  <img src="results/examples/BEANS0001.png" width="45%">
  <img src="results/examples/BEANS0037.png" width="45%">
</p>

<p align="center">
  <img src="results/examples/BEANS0063.png" width="45%">
  <img src="results/examples/BEANS0074.png" width="45%">
</p>

### Training Curves

![YOLO11s training curves](results/yolo11s_final/results.png)

### Validation Confusion Matrix

![Confusion matrix](results/yolo11s_final/confusion_matrix_normalized.png)

### Final Test Confusion Matrix

![Final test confusion matrix](results/final_test/confusion_matrix_normalized.png)

## Dataset

The project uses the 5K Groceries Object Detection Dataset, an extension
of the Freiburg Groceries Dataset.

Dataset characteristics:

- 4,947 images
- 25 grocery classes
- 11,663 annotated objects
- Pascal VOC bounding-box annotations
- 224×224 RGB images
- Average 2.36 objects per image

During dataset validation, one malformed zero-width bounding box was
identified and excluded during preprocessing.

### Split

A reproducible class-stratified split was created:

| Split | Images |
|---|---:|
| Train | 3,957 |
| Validation | 495 |
| Test | 495 |

The test set was kept untouched during model selection.

## Preprocessing

Pascal VOC annotations:

`[xmin, ymin, xmax, ymax]`

were converted to YOLO normalized format:

`[class_id, x_center, y_center, width, height]`

Converted labels were visually validated by transforming the normalized
coordinates back to pixel coordinates and overlaying them on the source
images.

## Baseline

The first model was a COCO-pretrained YOLO11n trained at the native
224×224 image resolution.

Validation results:

| Metric | YOLO11n |
|---|---:|
| Precision | 0.845 |
| Recall | 0.834 |
| mAP@0.50 | 0.892 |
| mAP@0.50:0.95 | 0.754 |

Training curves showed steadily decreasing train and validation losses
without obvious overfitting.

## Error Analysis

The normalized confusion matrix showed that most classes were strongly
separated, while some visually related categories produced more errors.

Examples included:

- flour / sugar
- oil / vinegar
- weaker performance on chocolate, soda, pasta, and nuts

The errors suggested that classification capacity was a more useful
next experiment than simply increasing input resolution.

## Model Improvement

Hypothesis:

A slightly larger detector may improve discrimination between visually
similar grocery classes.

The same training configuration was therefore repeated using YOLO11s.

| Metric | YOLO11n | YOLO11s |
|---|---:|---:|
| Precision | 0.845 | 0.872 |
| Recall | 0.834 | 0.879 |
| mAP@0.50 | 0.892 | 0.939 |
| mAP@0.50:0.95 | 0.754 | 0.820 |

YOLO11s was selected as the final model.

## Final Test Results

The selected model was evaluated once on the previously untouched
495-image test set containing 1,173 objects.

| Metric | Test Result |
|---|---:|
| Precision | 0.897 |
| Recall | 0.849 |
| mAP@0.50 | 0.917 |
| mAP@0.50:0.95 | 0.797 |

## Mixed-Class Co-occurrence Stress Test

The source dataset has an unusual property: every image contains objects from
only one semantic product class. Real retail scenes, however, often contain
many different product categories in the same field of view.

To test whether the final detector is sensitive to this difference, I created
a controlled synthetic co-occurrence stress test using the frozen YOLO11s
model.

For each of three seeds (42, 43, 44), I sampled a balanced held-out pool of
8 test images per class (200 source images). The same source images were then
composed into 448x448 2x2 mosaics under two conditions:

- **Same-class:** all four source images in a mosaic belonged to the same class.
- **Mixed-class:** all four source images belonged to different classes.

The original 224x224 images were not resized; they were placed directly into
the four quadrants. Therefore, within each seed, the two conditions used the
same source-image pool, object scale, canvas size, detector, and inference
settings. Only the class co-occurrence grouping changed.

### Results

| Seed | Condition | Precision | Recall | mAP@50 | mAP@50:95 |
|------|-----------|-----------|--------|--------|-----------|
| 42 | Same | 0.929 | 0.902 | 0.959 | 0.840 |
| 42 | Mixed | 0.875 | 0.826 | 0.907 | 0.791 |
| 43 | Same | 0.927 | 0.896 | 0.965 | 0.849 |
| 43 | Mixed | 0.864 | 0.833 | 0.921 | 0.807 |
| 44 | Same | 0.920 | 0.878 | 0.957 | 0.831 |
| 44 | Mixed | 0.869 | 0.790 | 0.898 | 0.774 |

Across the three seeds, mixed-class composition reduced:

- Precision by **0.056 ± 0.006**
- Recall by **0.076 ± 0.013**
- mAP@50 by **0.052 ± 0.008**
- mAP@50:95 by **0.049 ± 0.008**

The degradation was consistent across all three seeds. Recall showed the
largest average decline, indicating that mixed-class scenes primarily caused
more missed detections.

The largest mean per-class mAP@50:95 drops included flour (-0.145), nuts
(-0.128), pasta (-0.091), coffee (-0.082), and tea (-0.081). Several of these
classes were also among the weaker or more confused categories in the original
validation analysis.

### Interpretation

This experiment suggests that the detector is less robust when different
known product categories co-occur within the same image, a condition that is
not naturally represented in the source dataset.


## ONNX Export

The final PyTorch checkpoint was exported to ONNX.

PyTorch and ONNX inference were compared on the same image and produced:

- identical number of detections
- identical predicted classes
- matching confidence values
- matching bounding-box coordinates

This verified that model behavior was preserved after export.

## CPU Benchmark

Benchmark environment:

- Intel Core i5-1135G7
- CPU inference
- 224×224 input
- 100 test images
- warm-up before measurement

| Runtime | Latency | Throughput |
|---|---:|---:|
| PyTorch | 40.56 ms/image | 24.65 FPS |
| ONNX Runtime | 56.32 ms/image | 17.75 FPS |

ONNX Runtime was slower than PyTorch for this particular model and CPU
configuration. This demonstrates that model export does not
automatically guarantee faster inference and should be benchmarked on
the target hardware.

## API

The ONNX model is served through FastAPI.

Start the API:

```bash
uvicorn src.api:app --reload