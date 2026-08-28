from pathlib import Path
from ultralytics import YOLO

model = YOLO("models/best.pt")

metrics = model.val(
    data="configs/groceries.yaml",
    split="test",
    imgsz=224,
    batch=32,
    device="cpu",
    plots=True,
    project=str((Path.cwd() / "results").resolve()),
    name="final_test",
    exist_ok=True,
)

print("\nFinal test metrics")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

print("\nACTUAL SAVE DIRECTORY:")
print(metrics.save_dir)