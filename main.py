import os
import uuid
import time
from collections import defaultdict
from datetime import datetime, timezone
import boto3
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from inference import InferenceEngine
from visual_traits import analyze as analyze_traits

app = FastAPI(title="TrichAI API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine       = InferenceEngine("model/phytolens_v2.onnx")
START_TIME   = time.time()
VALID_LABELS = {"bud", "hash", "other", "plant"}
MAX_BYTES    = 10 * 1024 * 1024  # 10 MB


# ── MAX UPLOAD SIZE MIDDLEWARE (rejects oversized requests before reading body) ─
class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BYTES:
            return Response("Imagen demasiado grande. Máximo 10MB.", status_code=413)
        return await call_next(request)

app.add_middleware(MaxUploadSizeMiddleware)


# ── RATE LIMITING (20 req / 60 s per IP) ─────────────────────────────────────
RATE_LIMIT  = 20
RATE_WINDOW = 60.0
_rate_store: dict = defaultdict(list)

def _get_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown"
    )

def check_rate(request: Request):
    ip  = _get_ip(request)
    now = time.time()
    hits = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Demasiadas solicitudes. Espera un momento.")
    hits.append(now)
    _rate_store[ip] = hits


# ── IMAGE MAGIC BYTES VALIDATION ──────────────────────────────────────────────
def validate_image_magic(data: bytes):
    if data[:3] == b"\xff\xd8\xff":                        # JPEG
        return
    if data[:8] == b"\x89PNG\r\n\x1a\n":                  # PNG
        return
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":     # WebP
        return
    raise HTTPException(400, "El archivo no es una imagen válida.")


# ── STATS API KEY ─────────────────────────────────────────────────────────────
STATS_KEY = os.getenv("STATS_API_KEY", "")

def require_stats_key(request: Request):
    if not STATS_KEY:
        return
    if request.headers.get("x-api-key", "") != STATS_KEY:
        raise HTTPException(403, "Acceso denegado.")


# ── IN-MEMORY ANALYTICS ───────────────────────────────────────────────────────
_mem = {
    "total_analyses":   0,
    "by_label":         defaultdict(int),
    "total_contrib":    0,
    "contrib_by_label": defaultdict(int),
    "last_analysis":    None,
}

# ── UPSTASH REDIS (optional — persistent) ────────────────────────────────────
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

    top      = max(by_label, key=by_label.get) if by_label else None
    uptime_h = round((time.time() - START_TIME) / 3600, 1)

    return {
        "total_analyses":         total,
        "by_label":               by_label,
        "top_category":           top,
        "total_contributions":    total_c,
        "contributions_by_label": contrib_lbl,
        "last_analysis":          last,
        "uptime_hours":           uptime_h,
        "persistent":             redis is not None,
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
    for _label in VALID_LABELS:
        os.makedirs(os.path.join(CONTRIB_DIR, _label), exist_ok=True)


def save_contribution(data: bytes, label: str, ext: str) -> str:
    filename = f"{label}/{uuid.uuid4().hex}.{ext}"
    if USE_R2:
        s3.put_object(Bucket=R2_BUCKET, Key=filename, Body=data,
                      ContentType=f"image/{ext}")
    else:
        with open(os.path.join(CONTRIB_DIR, filename), "wb") as f:
            f.write(data)
    return filename


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     "trichai_v1",
        "storage":   "r2" if USE_R2 else "local",
        "analytics": "redis" if redis else "memory",
    }


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    check_rate(request)
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")
    validate_image_magic(contents)
    result = engine.predict(contents)
    result["visual_traits"] = analyze_traits(contents)
    record_analysis(result["label"])
    return {"success": True, "result": result}


@app.post("/contribute")
async def contribute(
    request: Request,
    file: UploadFile = File(...),
    label: str = Form(...),
):
    check_rate(request)
    if label not in VALID_LABELS:
        raise HTTPException(400, f"Etiqueta inválida. Usa: {', '.join(VALID_LABELS)}")
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")
    validate_image_magic(contents)
    ext = (file.filename or "photo.jpg").rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    saved = save_contribution(contents, label, ext)
    record_contribution(label)
    return {"success": True, "saved": saved, "label": label}


@app.get("/contribute/stats")
def contribute_stats(request: Request):
    require_stats_key(request)
    a     = get_analytics()
    stats = {label: a["contributions_by_label"].get(label, 0) for label in VALID_LABELS}
    stats["total"]   = a["total_contributions"]
    stats["storage"] = "r2" if USE_R2 else "local"
    return stats


@app.get("/stats")
def stats(request: Request):
    require_stats_key(request)
    return get_analytics()
