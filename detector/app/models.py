import torch.nn as nn
import torchvision.models as models

def get_detector_model(num_classes: int = 2, pretrained: bool = True):
	model = models.resnet18(pretrained=pretrained)
	in_features = model.fc.in_features
	model.fc = nn.Linear(in_features, num_classes)
	return model