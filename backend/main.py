# backend/main.py
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from colors import ColorConfig
from media import MediaError, cleanup_old, extract_frames, load_image_bytes, save_preview, sharpest_index
from service import ModelNotReady, analyze_images, model_info

# --- лимиты и пути ---
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024
PREVIEW_TTL_S = int(os.environ.get("PREVIEW_TTL_S", "3600"))   # 1 час

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Smart Washer AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    info = model_info()
    return {
        "status": "ok" if info.get("loaded") else "degraded",
        "model": info,
        "features": [
            "Fabric Classification (MobileNetV3-Large, TTA)",
            "Color Clash Detection (k-means + Lab)",
            "Video Frame Extraction (OpenCV, sharpest frames)",
            "Washing Mode Selection (washing.py)",
        ],
    }


@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    # --- 1. Валидация типа и размера ---
    if not file.content_type:
        raise HTTPException(400, "Не указан тип файла")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Пустой файл")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Файл больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ")

    is_video = "video" in file.content_type.lower()
    is_image = "image" in file.content_type.lower()
    if not (is_video or is_image):
        raise HTTPException(415, f"Неподдерживаемый тип: {file.content_type}")

    try:
        # --- 2. Кадры ---
        if is_video:
            images = extract_frames(data, n=5, max_side=640)
        else:
            images = [load_image_bytes(data, max_side=1280)]

        # --- 3. Анализ: ткань + цвет + режим (единая точка — service.py) ---
        result = analyze_images(images, gentle_on_color_mix=True, fast=False)

        # --- 4. Превью (для фото и видео — самый чёткий кадр) ---
        idx = sharpest_index(images)
        preview_name = save_preview(images[idx], UPLOAD_DIR, max_side=800)
        result["image_url"] = f"/uploads/{preview_name}"
        result["is_video"] = is_video
        result["frames_analyzed"] = len(images)

        # --- 5. Уборка старых превью (фото белья — приватные, хранить вечно незачем) ---
        cleanup_old(UPLOAD_DIR, max_age_s=PREVIEW_TTL_S)

        return result

    except MediaError as e:
        raise HTTPException(400, str(e))
    except ModelNotReady as e:
        # 503, а не 500: сервер жив, но модель не готова — фронт покажет «обучите модель»
        raise HTTPException(503, f"Модель не готова: {e}")
    except Exception as e:
        raise HTTPException(500, f"Ошибка анализа: {e}")