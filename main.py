import os
import uuid
import time
from collections import defaultdict
from datetime import datetime, timezone
import boto3
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import InferenceEngine

app = FastAPI(title="TrichAI API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine      = InferenceEngine("model/phytolens_v1.onnx")
START_TIME  = time.time()
VALID_LABELS = {"bud", "hash", "other", "plant"}

# ── IN-MEMORY ANALYTICS (persiste entre requests, se resetea al reiniciar) ──
# Si configuras Upstash Redis, se vuelve persistente automáticamente.
_mem = {
    "total_analyses":  0,
    "by_label":        defaultdict(int),
    "total_contrib":   0,
    "contrib_by_label":defaultdict(int),
    "last_analysis":   None,
}

# ── UPSTASH REDIS (opcional — persistente) ────────────────────────────────────
# Variables en Railway: UPSTASH_REDIS_URL
# Formato: rediss://:password@host:port
REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "")
redis = None
if REDIS_URL:
    try:
        import redis as redislib
        redis = redislib.from_url(REDIS_URL, decode_responses=True)
        redis.ping()
        print("Redis conectado — analytics persistentes")
    except Exception as e:
        print(f"Redis no disponible, usando memoria: {e}")
        redis = None


def _incr(key: str, amount: int = 1):
    if redis:
        redis.incrby(key, amount)

def _get(key: str, default=0):
    if redis:
        v = redis.get(key)
        return int(v) if v else default
    return default

def _hincr(hkey: str, field: str, amount: int = 1):
    if redis:
        redis.hincrby(hkey, field, amount)

def _hgetall(hkey: str):
    if redis:
        return {k: int(v) for k, v in redis.hgetall(hkey).items()}
    return {}


def record_analysis(label: str):
    _mem["total_analyses"] += 1
    _mem["by_label"][label] += 1
    _mem["last_analysis"] = datetime.now(timezone.utc).isoformat()
    _incr("trichai:total_analyses")
    _hincr("trichai:by_label", label)
    if redis:
        redis.set("trichai:last_analysis", _mem["last_analysis"])


def record_contribution(label: str):
    _mem["total_contrib"] += 1
    _mem["contrib_by_label"][label] += 1
    _incr("trichai:total_contrib")
    _hincr("trichai:contrib_by_label", label)


def get_analytics() -> dict:
    if redis:
        by_label    = _hgetall("trichai:by_label")
        contrib_lbl = _hgetall("trichai:contrib_by_label")
        total       = _get("trichai:total_analyses")
        total_c     = _get("trichai:total_contrib")
        last        = redis.get("trichai:last_analysis")
    else:
        by_label    = dict(_mem["by_label"])
        contrib_lbl = dict(_mem["contrib_by_label"])
        total       = _mem["total_analyses"]
        total_c     = _mem["total_contrib"]
        last        = _mem["last_analysis"]

    top = max(by_label, key=by_label.get) if by_label else None
    uptime_h = round((time.time() - START_TIME) / 3600, 1)

    return {
        "total_analyses":      total,
        "by_label":            by_label,
        "top_category":        top,
        "total_contributions": total_c,
        "contributions_by_label": contrib_lbl,
        "last_analysis":       last,
        "uptime_hours":        uptime_h,
        "persistent":          redis is not None,
    }


# ── R2 / S3 STORAGE ──────────────────────────────────────────────────────────
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
    CONTRIB_DIR = "contributions"
    os.makedirs(CONTRIB_DIR, exist_ok=True)
    for label in VALID_LABELS:
        os.makedirs(os.path.join(CONTRIB_DIR, label), exist_ok=True)


def save_contribution(data: bytes, label: str, ext: str) -> str:
    filename = f"{label}/{uuid.uuid4().hex}.{ext}"
    if USE_R2:
        s3.put_object(Bucket=R2_BUCKET, Key=filename, Body=data, ContentType=f"image/{ext}")
    else:
        with open(os.path.join(CONTRIB_DIR, filename), "wb") as f:
            f.write(data)
    return filename


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   "trichai_v1",
        "storage": "r2" if USE_R2 else "local",
        "analytics": "redis" if redis else "memory",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")
    result = engine.predict(contents)
    record_analysis(result["label"])
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
    ext = (file.filename or "photo.jpg").rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    saved = save_contribution(contents, label, ext)
    record_contribution(label)
    return {"success": True, "saved": saved, "label": label}


@app.get("/contribute/stats")
def contribute_stats():
    a = get_analytics()
    stats = {label: a["contributions_by_label"].get(label, 0) for label in VALID_LABELS}
    stats["total"]   = a["total_contributions"]
    stats["storage"] = "r2" if USE_R2 else "local"
    return stats


@app.get("/stats")
def stats():
    return get_analytics()
