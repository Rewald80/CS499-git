import os
import threading
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from io import BytesIO
from fastapi.responses import FileResponse



# --------------------------------------------
# App setup
# --------------------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")
os.makedirs("app/uploads", exist_ok=True)


@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    file_path = os.path.join("app/uploads", filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    # Disable caching
    return FileResponse(file_path, headers={"Cache-Control": "no-cache"})

# --------------------------------------------
# Upload results list
# --------------------------------------------
upload_results = []   # [{ "file": "image.jpg", "result": "Real" }]
MAX_HISTORY = 25

# --------------------------------------------
# Load model
# --------------------------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

checkpoint_path = "detector/app/model.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
else:
    print("⚠ WARNING: No model checkpoint found!")

model.eval()

# --------------------------------------------
# Image transform
# --------------------------------------------
model_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --------------------------------------------
# Federated Learning
# --------------------------------------------
def start_fl_server():
    import flwr as fl
    try:
        fl.server.start_server(
            server_address="127.0.0.1:8085",
            config={"num_rounds": 3}
        )
    except Exception as e:
        print("FL server error:", e)

# --------------------------------------------
# Routes
# --------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html.txt",
        {
            "request": request,
            "upload_results": upload_results
        }
    )

@app.post("/upload-image")
async def upload_image(request: Request, image: UploadFile = File(...)):
    try:
        # --- Save a display-sized version (256x256) ---
        contents = await image.read()
        img_display = Image.open(BytesIO(contents)).convert("RGB")
        img_display = img_display.resize((256, 256))
        save_path = os.path.join("app/uploads", image.filename)
        img_display.save(save_path, format="JPEG")

        # --- Prepare image for model ---
        img_model = Image.open(save_path).convert("RGB")
        img_tensor = model_transform(img_model).unsqueeze(0)

        # --- Run prediction ---
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
        result = "Deepfake" if pred == 1 else "Real"

        # --- Update upload history ---
        upload_results.insert(0, {"file": image.filename, "result": result})
        if len(upload_results) > MAX_HISTORY:
            upload_results.pop()

        return templates.TemplateResponse(
            "index.html.txt",
            {
                "request": request,
                "upload_results": upload_results
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/start-fl-server")
async def start_fl_training():
    thread = threading.Thread(target=start_fl_server, daemon=True)
    thread.start()
    return JSONResponse({"status": "Federated Learning training started!"})