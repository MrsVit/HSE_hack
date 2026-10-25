# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import uuid
from pathlib import Path
from model import get_classifier

app = FastAPI(title="Smart Washer AI", version="1.1")

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
        "input_size": "224x224",
        "num_classes": 7,
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Нужно изображение")

    file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}.{file_ext}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        classifier = get_classifier()
        result = classifier.analyze(str(file_path))
        result["image_url"] = f"/uploads/{file_path.name}"
        return result
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Ошибка анализа: {str(e)}")