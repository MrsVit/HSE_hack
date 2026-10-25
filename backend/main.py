from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
from pathlib import Path

from model import get_classifier
from media import extract_frames, load_image_bytes, MediaError
from colors import analyze_images, ColorConfig

app = FastAPI(title="Smart Washer AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MobileNetV3-Large (TextileClass-7)",
        "features": ["Fabric Classification", "Color Clash Detection", "Video Frame Extraction"],
        "input_size": "224x224",
        "num_classes": 7,
    }

@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(400, "Не указан тип файла")
    
    is_video = "video" in file.content_type.lower()
    file_bytes = await file.read()
    
    try:
        # 1. Извлечение кадров (1 для фото, 5 самых четких для видео)
        if is_video:
            images = extract_frames(file_bytes, n=5, max_side=640)
        else:
            images = [load_image_bytes(file_bytes, max_side=1280)]
            
        # 2. Анализ ткани (MobileNetV3-Large)
        classifier = get_classifier()
        fabric_result = classifier.analyze(images)
        
        # 3. Анализ цвета (предотвращение линьки)
        color_report = analyze_images(images, cfg=ColorConfig())
        
        # 4. Объединяем результаты
        result = {
            **fabric_result,
            "color_analysis": color_report.to_dict(),
            "is_video": is_video,
            "frames_analyzed": len(images),
            "image_url": f"/uploads/{save_temp_preview(images[0])}" if not is_video else None
        }
        return result
        
    except MediaError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ошибка анализа: {str(e)}")

def save_temp_preview(image) -> str:
    """Сохраняет превью для отображения на фронте"""
    name = f"{uuid.uuid4().hex}.jpg"
    image.convert("RGB").save(UPLOAD_DIR / name, "JPEG", quality=85)
    return name