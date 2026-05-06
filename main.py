import os
import uuid
import boto3
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import InferenceEngine

app = FastAPI(title="TrichAI API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine("model/phytolens_v1.onnx")

VALID_LABELS = {"bud", "hash", "other", "plant"}

# ── R2 / S3 storage ─────────────────────────────────────────────────────────
# Variables de entorno en Railway:
#   R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
R2_ENDPOINT   = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET     = os.getenv("R2_BUCKET", "trichai-contributions")

USE_R2 = all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY])

if USE_R2:
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
else:
    # Fallback: guardar en disco (se pierde al reiniciar Railway)
    CONTRIB_DIR = "contributions"
    os.makedirs(CONTRIB_DIR, exist_ok=True)
    for label in VALID_LABELS:
        os.makedirs(os.path.join(CONTRIB_DIR, label), exist_ok=True)


def save_contribution(data: bytes, label: str, ext: str) -> str:
    filename = f"{label}/{uuid.uuid4().hex}.{ext}"
    if USE_R2:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=filename,
            Body=data,
            ContentType=f"image/{ext}",
        )
    else:
        path = os.path.join(CONTRIB_DIR, filename)
        with open(path, "wb") as f:
            f.write(data)
    return filename


def count_contributions() -> dict:
    stats: dict = {label: 0 for label in VALID_LABELS}
    if USE_R2:
        for label in VALID_LABELS:
            paginator = s3.get_paginator("list_objects_v2")
            count = 0
            for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=f"{label}/"):
                count += page.get("KeyCount", 0)
            stats[label] = count
    else:
        for label in VALID_LABELS:
            folder = os.path.join(CONTRIB_DIR, label)
            stats[label] = len(os.listdir(folder)) if os.path.exists(folder) else 0
    stats["total"] = sum(stats.values())
    stats["storage"] = "r2" if USE_R2 else "local"
    return stats


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "trichai_v1", "storage": "r2" if USE_R2 else "local"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")
    result = engine.predict(contents)
    return {"success": True, "result": result}


@app.post("/contribute")
async def contribute(file: UploadFile = File(...), label: str = Form(...)):
    if label not in VALID_LABELS:
        raise HTTPException(400, f"Etiqueta inválida. Usa: {', '.join(VALID_LABELS)}")
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")

    ext      = (file.filename or "photo.jpg").rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"

    saved = save_contribution(contents, label, ext)
    return {"success": True, "saved": saved, "label": label, "storage": "r2" if USE_R2 else "local"}


@app.get("/contribute/stats")
def contribute_stats():
    return count_contributions()
