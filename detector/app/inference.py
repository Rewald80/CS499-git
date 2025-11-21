from pathlib import Path
import torch
from .model import get_detector_model
from PIL import Image
from torchvision import transforms

def load_model(checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_detector_model()
    
    # default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent / "model.pt"
    
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("Warning: No model checkpoint found. Model initialized randomly.")
    
    model.to(device)
    model.eval()
    return model, device

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image_pil(model, device, pil_image: Image.Image):
    img = eval_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        prob_fake = float(probs[0, 1].item())  # class 1 = fake
    return prob_fake
