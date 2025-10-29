import torch
from app.models import get_detector_model
from app.data import eval_transform
from PIL import Image
import io

def load_model(checkpoint_path: str = "model.pt", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = get_detector_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def predict_image_pil(model, device, pil_image: Image.Image):
    img = eval_transform(pil_image).unsqueeze(0).to(device)  # shape [1,3,H,W]
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        prob_fake = float(probs[0,1].item())  # class 1 = fake
    return prob_fake