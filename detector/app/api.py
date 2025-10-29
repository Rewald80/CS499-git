from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
from app.inference import load_model, predict_image_pil

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST","GET"], allow_headers=["*"])

MODEL_PATH = "model.pt"
model, device = load_model(MODEL_PATH)  # loads at startup

@app.get("/", response_class=HTMLResponse)
def read_root():
    return "<html><body><h2>Defaker API — POST /model</h2></body></html>"

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.post("/model")
async def run_model_with_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    prob_fake = predict_image_pil(model, device, pil_image)
    return {"probability_fake": prob_fake}