import os
import io
import torch
import torchvision
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image

app = FastAPI(title="Face Mask Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Fix: allow all origins, not just localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

# Fix: model is copied into /app by Dockerfile, so path is just the filename
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_detector High acccu.pth")

print(f"Loading weights from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    print(f"Files in /app: {os.listdir('/app')}")  # helpful debug

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

transform = transforms.Compose([transforms.ToTensor()])

label_map = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect"
}

@app.get("/")
def root():
    return {"status": "ok", "message": "Face Mask Detection API is running"}

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        boxes = prediction["boxes"].cpu().numpy().tolist()
        labels = prediction["labels"].cpu().numpy().tolist()
        scores = prediction["scores"].cpu().numpy().tolist()

        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                results.append({
                    "box": [round(b, 2) for b in box],
                    "label": label_map.get(label, "unknown"),
                    "score": round(score, 4)
                })

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)