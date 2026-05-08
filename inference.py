import onnxruntime as ort
import numpy as np
from PIL import Image
import io

CLASS_INFO = {
    0: {
        "label":      "bud",
        "display":    "Cogollo seco",
        "thc_min":    15,
        "thc_max":    30,
        "description": "Flor seca de cannabis, la forma más común de consumo.",
        "varieties":  ["OG Kush", "Amnesia Haze", "Purple Haze", "White Widow", "Gorilla Glue", "Gelato", "Fruta Prohibida", "Static"],
    },
    1: {
        "label":      "hash",
        "display":    "Hachís / Resina",
        "thc_min":    20,
        "thc_max":    60,
        "description": "Concentrado de resina de cannabis prensada.",
        "varieties":  ["Hash marroquí", "Charas", "Bubble hash", "Dry sift", "Hash libanés"],
    },
    2: {
        "label":      "other",
        "display":    "No detectado",
        "thc_min":    0,
        "thc_max":    0,
        "description": "No se ha detectado cannabis en la imagen.",
        "varieties":  [],
    },
    3: {
        "label":      "plant",
        "display":    "Planta viva",
        "thc_min":    10,
        "thc_max":    25,
        "description": "Planta de cannabis en fase de crecimiento o floración.",
        "varieties":  ["Cannabis sativa", "Cannabis indica", "Cannabis ruderalis", "Híbrido"],
    },
}

class InferenceEngine:
    def __init__(self, model_path: str):
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr  = (arr - mean) / std
        return arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, image_bytes: bytes) -> dict:
        tensor = self._preprocess(image_bytes)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        e      = np.exp(logits - logits.max())
        probs  = e / e.sum()
        idx    = int(probs.argmax())
        conf   = float(probs[idx])
        info   = CLASS_INFO[idx]

        thc_estimate = round(info["thc_min"] + (info["thc_max"] - info["thc_min"]) * conf)

        return {
            "label":        info["label"],
            "display":      info["display"],
            "confidence":   round(conf, 4),
            "quality":      "Alta" if conf>=0.85 else "Media" if conf>=0.65 else "Baja",
            "thc_min":      info["thc_min"],
            "thc_max":      info["thc_max"],
            "thc_estimate": thc_estimate,
            "description":  info["description"],
            "varieties":    info["varieties"],
            "all_probs":    {CLASS_INFO[i]["label"]: round(float(p),4) for i,p in enumerate(probs)},
        }