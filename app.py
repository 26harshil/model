# import os
# import io
# import torch
# import torchvision
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from torchvision import transforms
# from PIL import Image

# app = FastAPI(title="Face Mask Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Fix: allow all origins, not just localhost
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_classes = 4
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
#     in_features, num_classes
# )

# # Fix: model is copied into /app by Dockerfile, so path is just the filename
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_detector High acccu.pth")

# print(f"Loading weights from {MODEL_PATH}...")
# if not os.path.exists(MODEL_PATH):
#     print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
#     print(f"Files in /app: {os.listdir('/app')}")  # helpful debug

# try:
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")

# transform = transforms.Compose([transforms.ToTensor()])

# label_map = {
#     1: "with_mask",
#     2: "without_mask",
#     3: "mask_weared_incorrect"
# }

# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Face Mask Detection API is running"}

# @app.post("/predict")
# async def predict_mask(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         img_tensor = transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             prediction = model(img_tensor)[0]

#         boxes = prediction["boxes"].cpu().numpy().tolist()
#         labels = prediction["labels"].cpu().numpy().tolist()
#         scores = prediction["scores"].cpu().numpy().tolist()

#         results = []
#         for box, label, score in zip(boxes, labels, scores):
#             if score > 0.5:
#                 results.append({
#                     "box": [round(b, 2) for b in box],
#                     "label": label_map.get(label, "unknown"),
#                     "score": round(score, 4)
#                 })

#         return JSONResponse(content={"predictions": results})

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)
import os
import io
import torch
import torchvision
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image

# ==============================
# ⚙️ CONFIG
# ==============================
CONFIDENCE_THRESHOLD = 0.5
MAX_IMAGE_SIZE = 1024  # prevent very large images (performance)
MODEL_FILENAME = "mask_detector High acccu.pth"

# ==============================
# 🚀 APP INIT
# ==============================
app = FastAPI(title="Face Mask Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (frontend access)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# ⚡ DEVICE SETUP
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimize CPU usage (important for HuggingFace)
torch.set_num_threads(1)

# ==============================
# 🧠 MODEL LOAD
# ==============================
num_classes = 4

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    MODEL_FILENAME
)

print(f"📦 Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    print(f"Files in directory: {os.listdir(os.path.dirname(MODEL_PATH))}")

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==============================
# 🔄 TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.ToTensor()
])

# ==============================
# 🏷️ LABEL MAP
# ==============================
label_map = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect"
}

# ==============================
# ❤️ HEALTH CHECK
# ==============================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Face Mask Detection API is running 🚀"
    }

# ==============================
# 🔍 PREDICT ENDPOINT
# ==============================
@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    try:
        # 📥 Read Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 📏 Resize if too large (performance boost)
        width, height = image.size
        if max(width, height) > MAX_IMAGE_SIZE:
            image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

        resized_width, resized_height = image.size

        # 🔄 Transform
        img_tensor = transform(image).unsqueeze(0).to(device)

        # 🧠 Prediction
        with torch.no_grad():
            prediction = model(img_tensor)[0]

        boxes = prediction["boxes"].cpu().numpy().tolist()
        labels = prediction["labels"].cpu().numpy().tolist()
        scores = prediction["scores"].cpu().numpy().tolist()

        # 📊 Filter results
        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score > CONFIDENCE_THRESHOLD:
                results.append({
                    "box": [round(b, 2) for b in box],
                    "label": label_map.get(label, "unknown"),
                    "score": round(score, 4)
                })

        # 📤 Response
        return JSONResponse(content={
            "predictions": results,
            "image_size": {
                "width": resized_width,
                "height": resized_height
            },
            "threshold": CONFIDENCE_THRESHOLD,
            "count": len(results)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# ==============================
# 🚀 RUN (LOCAL)
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)