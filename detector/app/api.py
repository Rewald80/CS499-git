import os
import threading
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from io import BytesIO

# --------------------------------------------
# App setup
# --------------------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")
os.makedirs("app/uploads", exist_ok=True)

# --------------------------------------------
# Display uploaded images without caching
# --------------------------------------------
@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    file_path = os.path.join("app/uploads", filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path, headers={"Cache-Control": "no-cache"})

# --------------------------------------------
# Upload results history
# --------------------------------------------
upload_results = []  
MAX_HISTORY = 25

# --------------------------------------------
# Load Model
# --------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nLoading ResNet18 model...")
model = models.resnet18(weights=None)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 2)
)

checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pt")
if os.path.exists(checkpoint_path):
    print(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("⚠ WARNING: No model checkpoint found!")

model = model.to(device, memory_format=torch.channels_last)
model.eval()

# --------------------------------------------
# Image Transform
# --------------------------------------------
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------
# Federated Learning Server
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


@app.post("/start-fl-server")
async def start_fl_training():
    thread = threading.Thread(target=start_fl_server, daemon=True)
    thread.start()
    return JSONResponse({"status": "Federated Learning training started!"})

# --------------------------------------------
# ROUTES
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

# --------------------------------------------
# UPLOAD & DETECT DEEPFAKE
# --------------------------------------------
@app.post("/upload-image")
async def upload_image(request: Request, image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_raw = Image.open(BytesIO(contents)).convert("RGB")
        img_display = img_raw.resize((256, 256))
        save_path = os.path.join("app/uploads", image.filename)
        img_display.save(save_path, format="JPEG")
        img_tensor = model_transform(img_raw).unsqueeze(0)
        img_tensor = img_tensor.to(device, memory_format=torch.channels_last)

        # ---- Model Prediction ----
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()

        result = "Deepfake" if pred == 0 else "Real"

        # ---- Update history ----
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
