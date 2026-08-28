from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "models/best.pt"
IMAGE_DIR = Path("data/processed/yolo/images/test")
OUTPUT_DIR = Path("results/examples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

# Fixed examples so the README is reproducible
sample_images = sorted(IMAGE_DIR.glob("*"))[:6]

for image_path in sample_images:
    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.25,
        device="cpu",
        verbose=False,
        save=False,
    )

    annotated = results[0].plot()

    output_path = OUTPUT_DIR / image_path.name

    import cv2
    cv2.imwrite(str(output_path), annotated)

    print("Saved:", output_path)