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

# Setup CORS so the frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allows the Next.js app to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Recreate the exact model architecture from your notebook
num_classes = 4
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

# 2. Load your custom trained weights
# Construct absolute path to the model file in the parent directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector High acccu.pth")

print(f"Loading weights from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: Model file '{MODEL_PATH}' not found!")

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the transform (same as in your Dataset class)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Reverse mapping for your labels to make the API output human-readable
label_map = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect"
}

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    try:
        # Read the image file from the user
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Transform image to tensor and add a batch dimension
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            prediction = model(img_tensor)[0]
            
        # Parse the outputs
        boxes = prediction["boxes"].cpu().numpy().tolist()
        labels = prediction["labels"].cpu().numpy().tolist()
        scores = prediction["scores"].cpu().numpy().tolist()
        
        results = []
        # We use a 0.5 confidence threshold, just like your evaluation loop
        confidence_threshold = 0.5 
        
        for box, label, score in zip(boxes, labels, scores):
            if score > confidence_threshold:
                results.append({
                    "box": [round(box[0], 2), round(box[1], 2), round(box[2], 2), round(box[3], 2)],
                    "label": label_map.get(label, "unknown"),
                    "score": round(score, 4)
                })
                
        # Send predictions in the format the frontend expects!
        return JSONResponse(content={"predictions": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)