# Retail Product Detection

A 25-class grocery product detector built with YOLO11 on the 5K Groceries
dataset. Trained, evaluated, stress-tested, exported to ONNX, and served
through FastAPI.

## Problem

Retail computer vision needs to detect and identify products across
different angles, occlusion, and packaging. This project builds a 25-class
grocery detector and covers the full pipeline: dataset validation, training,
error analysis, model comparison, a stress test on the dataset's main
limitation, ONNX export, CPU benchmarking, and a FastAPI service.

## Example Predictions

<p align="center">
  <img src="results/examples/BEANS0001.png" width="45%">
  <img src="results/examples/BEANS0037.png" width="45%">
</p>
<p align="center">
  <img src="results/examples/BEANS0063.png" width="45%">
  <img src="results/examples/BEANS0074.png" width="45%">
</p>

**Training curves**

![YOLO11s training curves](results/yolo11s_final/results.png)

**Final test confusion matrix**

![Final test confusion matrix](results/final_test/confusion_matrix_normalized.png)

## Dataset

5K Groceries Object Detection Dataset (extends the Freiburg Groceries
Dataset).

- 4,947 images, 224×224 RGB
- 25 classes, 11,663 annotated objects (avg. 2.36 objects/image)
- Pascal VOC annotations, converted to YOLO format and checked by re-plotting
  boxes on the source images
- One malformed zero-width box found and excluded

Class-stratified 80/10/10 split, seed 42:

| Split | Images |
|---|---:|
| Train | 3,957 |
| Validation | 495 |
| Test | 495 |

Test set was kept untouched until the final model was chosen.

## Training & Results

Started with a COCO-pretrained YOLO11n at 224×224. Validation loss dropped
steadily with no obvious overfitting, but the confusion matrix showed some
visually similar classes mixed up (flour/sugar, oil/vinegar) plus weaker
scores on chocolate, soda, pasta, and nuts. This looked more like a
model-capacity limit than a resolution limit, so I reran the same setup on
YOLO11s.

| Metric | YOLO11n (val) | YOLO11s (val) | YOLO11s (test) |
|---|---:|---:|---:|
| Precision | 0.845 | 0.872 | 0.897 |
| Recall | 0.834 | 0.879 | 0.849 |
| mAP@0.50 | 0.892 | 0.939 | 0.917 |
| mAP@0.50:0.95 | 0.754 | 0.820 | 0.797 |

YOLO11s improved across every metric and became the final model. Test
results are from one evaluation on the untouched 495-image test set
(1,173 objects).

## Mixed-Class Co-occurrence Stress Test

Every image in this dataset contains objects from only one product class.
Real shelves and carts don't look like that; products sit next to each
other. I tested whether that gap actually hurts the model, instead of
assuming it does.

Using the frozen YOLO11s model, I composed held-out test images into 448×448
2×2 mosaics two ways: four images from the same class, or four images from
different classes. Everything else (image pool, object scale, canvas size,
model, inference settings) stayed the same between the two, and I repeated
it across three seeds (42, 43, 44) with 8 images per class per seed.

Mixed-class scenes scored consistently worse across all three seeds:

| Metric | Drop (mixed vs. same) |
|---|---:|
| Precision | 0.056 ± 0.006 |
| Recall | 0.076 ± 0.013 |
| mAP@50 | 0.052 ± 0.008 |
| mAP@50:95 | 0.049 ± 0.008 |

Recall dropped the most, so the model mostly missed objects rather than
mislabeling them. The classes with the biggest drop (flour, nuts, pasta,
coffee, tea) overlap with the classes that were already weak in the
original error analysis.

<details>
<summary>Per-seed results</summary>

| Seed | Condition | Precision | Recall | mAP@50 | mAP@50:95 |
|------|-----------|-----------|--------|--------|-----------|
| 42 | Same | 0.929 | 0.902 | 0.959 | 0.840 |
| 42 | Mixed | 0.875 | 0.826 | 0.907 | 0.791 |
| 43 | Same | 0.927 | 0.896 | 0.965 | 0.849 |
| 43 | Mixed | 0.864 | 0.833 | 0.921 | 0.807 |
| 44 | Same | 0.920 | 0.878 | 0.957 | 0.831 |
| 44 | Mixed | 0.869 | 0.790 | 0.898 | 0.774 |

</details>

This is a synthetic test, not real shelf footage, so it doesn't prove
real-world performance. What it does show is that the model is less robust
to a scenario the training data never represents.

## Deployment

Exported the final PyTorch checkpoint to ONNX and checked it against the
original model on the same image: same detections, same classes, same
confidence scores, same boxes. Export didn't change model behavior.

Benchmarked both runtimes on CPU (Intel i5-1135G7, 224×224 input, 100 test
images, with warm-up):

| Runtime | Latency | Throughput |
|---|---:|---:|
| PyTorch | 40.56 ms/image | 24.65 FPS |
| ONNX Runtime | 56.32 ms/image | 17.75 FPS |

ONNX Runtime was slower here, not faster. Exporting a model doesn't
automatically speed it up, and it's worth benchmarking on the actual target
hardware before assuming otherwise.

The ONNX model is served through FastAPI:

```bash
uvicorn src.api:app --reload
```