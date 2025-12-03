import sys
import os
import glob
import numpy as np
import torch
import cv2
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import MultiTaskResNet
from src.explain import GradCAM, generate_occlusion_map

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Global variables
model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "cache"
CACHE_FILES = []

def load_model(model_path="model_clean_serie1.pth"):
    global model
    model = MultiTaskResNet().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")

@app.on_event("startup")
async def startup_event():
    load_model()
    global CACHE_FILES
    CACHE_FILES = glob.glob(os.path.join(CACHE_DIR, "*_X.npy"))
    if not CACHE_FILES:
        print("Warning: No cache files found.")

class PredictResponse(BaseModel):
    shot_prob: list
    stress_val: float
    task_prob: list
    image_id: str

@app.get("/api/random_sample")
async def get_random_sample():
    if not CACHE_FILES:
        raise HTTPException(status_code=404, detail="No cache files found")
    
    # Pick a random file
    f = np.random.choice(CACHE_FILES)
    # Load the array
    X = np.load(f) # (N, 5, 64, 64)
    # Pick a random sample
    idx = np.random.randint(0, len(X))
    sample = X[idx]
    
    # Encode as base64 for visualization (just showing the first band - Delta)
    # Normalize to 0-255
    img_disp = sample[0]
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
    img_disp = (img_disp * 255).astype(np.uint8)
    img_disp = cv2.applyColorMap(img_disp, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode('.png', img_disp)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "image_id": f"{os.path.basename(f)}:{idx}",
        "image_base64": img_str
    }

@app.post("/api/predict")
async def predict(data: dict):
    image_id = data.get("image_id")
    if not image_id:
        raise HTTPException(status_code=400, detail="image_id required")
    
    fname, idx = image_id.split(":")
    idx = int(idx)
    fpath = os.path.join(CACHE_DIR, fname)
    
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
        
    X = np.load(fpath)
    sample = X[idx]
    x_tensor = torch.tensor(sample).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        pred_shot, pred_stress, pred_task = model(x_tensor)
        
    return {
        "shot_prob": torch.softmax(pred_shot, dim=1).cpu().numpy().tolist()[0],
        "stress_val": pred_stress.item(),
        "task_prob": torch.softmax(pred_task, dim=1).cpu().numpy().tolist()[0]
    }

@app.post("/api/explain")
async def explain(data: dict):
    image_id = data.get("image_id")
    method = data.get("method", "gradcam") # gradcam or occlusion
    head = data.get("head", "task")
    
    fname, idx = image_id.split(":")
    idx = int(idx)
    fpath = os.path.join(CACHE_DIR, fname)
    
    X = np.load(fpath)
    sample = X[idx]
    x_tensor = torch.tensor(sample).unsqueeze(0).float().to(DEVICE)
    x_tensor.requires_grad = True
    
    if method == "gradcam":
        # Target layer: last conv layer of layer4
        target_layer = model.backbone.layer4[-1].conv2
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(x_tensor, head_name=head)
        
        # Visualize
        cam = (cam * 255).astype(np.uint8)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.png', cam)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {"image_base64": img_str}
        
    elif method == "occlusion":
        occ = generate_occlusion_map(model, x_tensor, head_name=head)
        
        # Visualize
        occ = (occ * 255).astype(np.uint8)
        occ = cv2.applyColorMap(occ, cv2.COLORMAP_HOT)
        _, buffer = cv2.imencode('.png', occ)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {"image_base64": img_str}
    
    else:
        raise HTTPException(status_code=400, detail="Unknown method")
