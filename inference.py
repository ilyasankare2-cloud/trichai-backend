import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

CLASS_INFO = {
    0: {
        "label":    "bud",
        "display":  "Cogollo seco",
        "thc_min":  15,
        "thc_max":  30,
        "description": "Flor seca de cannabis, la forma más común de consumo.",
        "varieties": ["OG Kush", "Amnesia Haze", "Purple Haze", "White Widow", "Gorilla Glue", "Gelato", "Forbidden Fruit", "Static"],
    },
    1: {
        "label":    "hash",
        "display":  "Hachís / Resina",
        "thc_min":  20,
        "thc_max":  60,
        "description": "Concentrado de resina de cannabis prensada.",
        "varieties": ["Hash marroquí", "Charas", "Bubble hash", "Dry sift", "Lebanese hash"],
    },
    2: {
        "label":    "other",
        "display":  "Otro producto",
        "thc_min":  5,
        "thc_max":  90,
        "description": "Extracto, aceite u otro derivado del cannabis.",
        "varieties": ["BHO", "Rosin", "Wax", "Shatter", "Aceite CBD", "Tintura"],
    },
    3: {
        "label":    "plant",
        "display":  "Planta viva",
        "thc_min":  10,
        "thc_max":  25,
        "description": "Planta de cannabis en fase de crecimiento o floración.",
        "varieties": ["Cannabis sativa", "Cannabis indica", "Cannabis ruderalis", "Híbrido"],
    },
}

class InferenceEngine:
    def __init__(self, model_path: str):
        model = models.efficientnet_v2_s(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256),
            nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 4),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        self.model = model
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def predict(self, image_bytes: bytes) -> dict:
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.tf(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor)[0], dim=0).numpy()
        idx  = int(probs.argmax())
        conf = float(probs[idx])
        info = CLASS_INFO[idx]

        # Estimar THC basado en confianza
        thc_estimate = round(
            info["thc_min"] + (info["thc_max"] - info["thc_min"]) * conf
        )

        return {
            "label":       info["label"],
            "display":     info["display"],
            "confidence":  round(conf, 4),
            "quality":     "Alta" if conf>=0.85 else "Media" if conf>=0.65 else "Baja",
            "thc_min":     info["thc_min"],
            "thc_max":     info["thc_max"],
            "thc_estimate":thc_estimate,
            "description": info["description"],
            "varieties":   info["varieties"],
            "all_probs":   {CLASS_INFO[i]["label"]: round(float(p),4) for i,p in enumerate(probs)},
        }