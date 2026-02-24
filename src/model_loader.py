import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]  # remonte à bjj_stats/
MODEL_PATH = BASE_DIR / "models" / "bjj_model_best.pth"


def load_model():
    device = torch.device("cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    classes = checkpoint["classes"]
    model_state = checkpoint["model_state"]

    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(classes))

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, classes, checkpoint