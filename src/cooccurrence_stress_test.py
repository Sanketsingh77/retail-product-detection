from pathlib import Path
import random
import shutil

import pandas as pd
import yaml
from PIL import Image
from ultralytics import YOLO


ROOT = Path.cwd()

TEST_IMAGES = ROOT / "data/processed/yolo/images/test"
TEST_LABELS = ROOT / "data/processed/yolo/labels/test"
MODEL_PATH = ROOT / "models/best.pt"

OUT_ROOT = ROOT / "results/cooccurrence_stress_test"

SEEDS = [42, 43, 44]

SOURCE_SIZE = 224
CANVAS_SIZE = 448
QUADRANTS = [
    (0, 0),
    (224, 0),
    (0, 224),
    (224, 224),
]

# Balanced source pool per class.
# 12 × 25 classes = 300 source images -> 75 mosaics per condition.
TARGET_PER_CLASS = 12


with open(ROOT / "configs/groceries.yaml", "r") as f:
    base_yaml = yaml.safe_load(f)

NAMES = {int(k): v for k, v in base_yaml["names"].items()}


def image_class(image_path):
    label_path = TEST_LABELS / f"{image_path.stem}.txt"

    lines = [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    classes = {int(line.split()[0]) for line in lines}

    # Dataset property already established in exploration:
    # one semantic class per source image.
    assert len(classes) == 1

    return next(iter(classes))


def transform_labels(image_path, x_offset, y_offset):
    """
    Translate YOLO labels from a 224x224 source image into
    one quadrant of a 448x448 canvas.
    """
    label_path = TEST_LABELS / f"{image_path.stem}.txt"
    output = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = line.split()

        cls = int(cls)
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        # 224 pixels occupy half of the 448 canvas.
        new_x = (x * 0.5) + (x_offset / CANVAS_SIZE)
        new_y = (y * 0.5) + (y_offset / CANVAS_SIZE)
        new_w = w * 0.5
        new_h = h * 0.5

        assert 0 <= new_x <= 1
        assert 0 <= new_y <= 1
        assert 0 < new_w <= 1
        assert 0 < new_h <= 1

        output.append(
            f"{cls} {new_x:.6f} {new_y:.6f} "
            f"{new_w:.6f} {new_h:.6f}"
        )

    return output


def build_mosaic(group, output_image, output_label):
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE))

    labels = []

    for image_path, (x_offset, y_offset) in zip(group, QUADRANTS):
        image = Image.open(image_path).convert("RGB")

        assert image.size == (SOURCE_SIZE, SOURCE_SIZE)

        canvas.paste(image, (x_offset, y_offset))

        labels.extend(
            transform_labels(
                image_path,
                x_offset,
                y_offset,
            )
        )

    canvas.save(output_image)
    output_label.write_text("\n".join(labels) + "\n")


def make_dataset(seed):
    rng = random.Random(seed)

    image_paths = sorted(
        p
        for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    by_class = {c: [] for c in NAMES}

    for image_path in image_paths:
        c = image_class(image_path)
        by_class[c].append(image_path)

    min_available = min(len(v) for v in by_class.values())

    n_per_class = min(TARGET_PER_CLASS, min_available)
    n_per_class -= n_per_class % 4

    if n_per_class < 4:
        raise RuntimeError("Not enough images per class.")

    selected = {}

    for c, images in by_class.items():
        images = images.copy()
        rng.shuffle(images)
        selected[c] = images[:n_per_class]

    # ---------------------------------------------------------
    # SAME-CLASS GROUPS
    # ---------------------------------------------------------

    same_groups = []

    for c in sorted(selected):
        images = selected[c]

        for i in range(0, n_per_class, 4):
            same_groups.append(images[i:i + 4])

    rng.shuffle(same_groups)

    # ---------------------------------------------------------
    # MIXED-CLASS GROUPS
    #
    # Uses exactly the SAME source images as same-class.
    # Interleaving ensures four consecutive items come from
    # four different classes.
    # ---------------------------------------------------------

    class_order = sorted(selected)
    rng.shuffle(class_order)

    mixed_sequence = []

    for i in range(n_per_class):
        for c in class_order:
            mixed_sequence.append(selected[c][i])

    mixed_groups = [
        mixed_sequence[i:i + 4]
        for i in range(0, len(mixed_sequence), 4)
    ]

    # Cheap automatic methodological check.
    for group in mixed_groups:
        classes = [image_class(p) for p in group]
        assert len(set(classes)) == 4

    assert len(same_groups) == len(mixed_groups)

    seed_root = OUT_ROOT / f"seed_{seed}"

    for condition, groups in [
        ("same", same_groups),
        ("mixed", mixed_groups),
    ]:
        condition_root = seed_root / condition
        image_dir = condition_root / "images"
        label_dir = condition_root / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx, group in enumerate(groups):
            build_mosaic(
                group,
                image_dir / f"mosaic_{idx:04d}.png",
                label_dir / f"mosaic_{idx:04d}.txt",
            )

        dataset_yaml = {
            "path": str(condition_root.resolve()),
            "train": "images",
            "val": "images",
            "test": "images",
            "names": NAMES,
        }

        with open(condition_root / "dataset.yaml", "w") as f:
            yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print(
        f"Seed {seed}: "
        f"{len(same_groups)} same + "
        f"{len(mixed_groups)} mixed mosaics "
        f"using {n_per_class} images/class"
    )


def evaluate():
    model = YOLO(str(MODEL_PATH))

    summary_rows = []
    per_class_rows = []

    for seed in SEEDS:
        for condition in ["same", "mixed"]:

            dataset_yaml = (
                OUT_ROOT
                / f"seed_{seed}"
                / condition
                / "dataset.yaml"
            )

            print(f"\nEvaluating seed={seed}, condition={condition}")

            metrics = model.val(
                data=str(dataset_yaml),
                split="test",
                imgsz=448,
                batch=16,
                device="cpu",
                workers=2,
                plots=True,
                project=str(OUT_ROOT / "evaluation"),
                name=f"seed_{seed}_{condition}",
                exist_ok=True,
                verbose=False,
            )

            summary_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                }
            )

            class_maps = metrics.box.maps

            for class_id, class_map in enumerate(class_maps):
                per_class_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "class_id": class_id,
                        "class_name": NAMES[class_id],
                        "map50_95": float(class_map),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)

    summary.to_csv(
        OUT_ROOT / "summary.csv",
        index=False,
    )

    per_class.to_csv(
        OUT_ROOT / "per_class.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # OVERALL DELTA: MIXED - SAME
    # ---------------------------------------------------------

    pivot = summary.pivot(
        index="seed",
        columns="condition",
        values=[
            "precision",
            "recall",
            "map50",
            "map50_95",
        ],
    )

    delta_rows = []

    for seed in SEEDS:
        row = {"seed": seed}

        for metric in [
            "precision",
            "recall",
            "map50",
            "map50_95",
        ]:
            same = pivot.loc[seed, (metric, "same")]
            mixed = pivot.loc[seed, (metric, "mixed")]

            row[f"{metric}_same"] = same
            row[f"{metric}_mixed"] = mixed
            row[f"{metric}_delta"] = mixed - same

        delta_rows.append(row)

    deltas = pd.DataFrame(delta_rows)

    deltas.to_csv(
        OUT_ROOT / "deltas_by_seed.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # PER-CLASS DELTAS
    # ---------------------------------------------------------

    class_pivot = per_class.pivot(
        index=["seed", "class_id", "class_name"],
        columns="condition",
        values="map50_95",
    ).reset_index()

    class_pivot["delta_mixed_minus_same"] = (
        class_pivot["mixed"] - class_pivot["same"]
    )

    class_pivot.to_csv(
        OUT_ROOT / "per_class_deltas_by_seed.csv",
        index=False,
    )

    class_summary = (
        class_pivot
        .groupby(["class_id", "class_name"])["delta_mixed_minus_same"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )

    class_summary.to_csv(
        OUT_ROOT / "per_class_delta_summary.csv",
        index=False,
    )

    print("\n==============================")
    print("OVERALL RESULTS")
    print("==============================")

    print(summary.to_string(index=False))

    print("\n==============================")
    print("MIXED - SAME DELTAS")
    print("==============================")

    print(
        deltas[
            [
                "seed",
                "precision_delta",
                "recall_delta",
                "map50_delta",
                "map50_95_delta",
            ]
        ].to_string(index=False)
    )

    print("\nMean delta across seeds:")

    for metric in [
        "precision_delta",
        "recall_delta",
        "map50_delta",
        "map50_95_delta",
    ]:
        print(
            f"{metric}: "
            f"{deltas[metric].mean():.4f} "
            f"(std {deltas[metric].std():.4f})"
        )

    print("\nMost negatively affected classes:")

    print(
        class_summary.head(10).to_string(index=False)
    )


if __name__ == "__main__":

    # Only removes this NEW experiment's previous output.
    # Existing baseline/final/test results are untouched.
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True)

    for seed in SEEDS:
        make_dataset(seed)

    evaluate()