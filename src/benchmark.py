import time
from pathlib import Path

from ultralytics import YOLO


PT_MODEL = Path("models/best.pt")
ONNX_MODEL = Path("models/best.onnx")
IMAGE_DIR = Path("data/processed/yolo/images/test")


pt_model = YOLO(PT_MODEL)
onnx_model = YOLO(ONNX_MODEL)

images = list(IMAGE_DIR.glob("*"))[:100]

print("Benchmark images:", len(images))


def benchmark(model, images, warmup=10):
    for image in images[:warmup]:
        model.predict(
            source=str(image),
            imgsz=224,
            device="cpu",
            verbose=False,
        )

    start = time.perf_counter()

    for image in images:
        model.predict(
            source=str(image),
            imgsz=224,
            device="cpu",
            verbose=False,
        )

    end = time.perf_counter()

    total_time = end - start
    latency = total_time / len(images)

    return {
        "total_seconds": total_time,
        "latency_ms": latency * 1000,
        "fps": 1 / latency,
    }


print("\nPyTorch CPU")
pt_results = benchmark(pt_model, images)
print(pt_results)

print("\nONNX Runtime CPU")
onnx_results = benchmark(onnx_model, images)
print(onnx_results)