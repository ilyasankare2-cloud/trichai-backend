import numpy as np
from PIL import Image
import io


def analyze(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img.resize((128, 128), Image.LANCZOS), dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Brightness 0-100
    brightness = float(arr.mean() / 255 * 100)

    # Texture roughness: normalized std dev scaled to 0-100
    roughness = min(float(arr.std() / 128 * 100), 100)

    # Per-pixel HSV saturation
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)
    saturation = float(sat.mean() * 100)

    # Trichome coverage: bright pixels with low saturation (white/silver)
    bright   = arr.mean(axis=2) > 200
    low_sat  = sat < 0.25
    trichome_pct = float((bright & low_sat).mean() * 100)

    # Dominant color
    avg_rgb = [int(r.mean()), int(g.mean()), int(b.mean())]

    # Warmth: red dominance over blue (golden/brown tones = well cured)
    warmth = float(max(0.0, float(r.mean()) - float(b.mean())) / 255 * 100)

    # Labels
    if roughness > 55:
        texture = "Muy cristalina"
    elif roughness > 35:
        texture = "Cristalina"
    elif roughness > 20:
        texture = "Media"
    else:
        texture = "Lisa"

    if trichome_pct > 12:
        trichomes = "Alta"
    elif trichome_pct > 4:
        trichomes = "Media"
    else:
        trichomes = "Baja"

    if brightness < 30:
        cure = "Muy oscura"
    elif brightness > 75:
        cure = "Muy seca"
    elif warmth > 15:
        cure = "Bien curada"
    else:
        cure = "Fresca"

    # Hash-specific derived fields (previously computed client-side, see TD-015).
    if roughness < 25:
        uniformity = "Uniforme"
    elif roughness < 45:
        uniformity = "Media"
    else:
        uniformity = "Irregular"

    gr, gg, gb = avg_rgb
    green_tint = gg > gr and gg > gb + 10

    return {
        "brightness":        round(brightness, 1),
        "roughness":         round(roughness, 1),
        "texture":           texture,
        "saturation":        round(saturation, 1),
        "trichome_coverage": round(trichome_pct, 1),
        "trichomes":         trichomes,
        "dominant_color":    avg_rgb,
        "cure":              cure,
        "warmth":            round(warmth, 1),
        "uniformity":        uniformity,
        "green_tint":        green_tint,
    }
