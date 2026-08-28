from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO


app = FastAPI(
    title="Retail Product Detection API",
    version="1.0"
)

model = YOLO("models/best.onnx")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "YOLO11s ONNX"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    result = model.predict(
        source=image,
        imgsz=224,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": round(confidence, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    return {
        "num_detections": len(detections),
        "detections": detections
    }