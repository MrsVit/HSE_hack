# backend/media.py
"""Кадры из видео и захват с камеры, подключённой к самому устройству (серверу).

Видео из браузера (MediaRecorder) приходит как webm/mp4 без нормальных метаданных о числе кадров,
поэтому читаем его последовательно, а не по индексам. Из всего ролика берём N самых чётких кадров,
по одному из каждого равного отрезка: размытые кадры при вращении барабана вредят классификатору,
а равномерная выборка даёт разные вещи в разных ракурсах.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import List, Union

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:                      # без OpenCV работают только фото
    cv2 = None


class MediaError(ValueError):
    """Ошибка, которую можно показать пользователю как есть."""


def _need_cv2() -> None:
    if cv2 is None:
        raise MediaError("Для видео нужен OpenCV: pip install opencv-python-headless")


def sharpness(bgr: np.ndarray) -> float:
    """Дисперсия лапласиана: чем больше, тем чётче кадр."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _shrink(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return bgr


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pick_sharpest(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """Делит кадры на n равных отрезков и берёт самый чёткий из каждого (порядок по времени сохраняется)."""
    if len(frames) <= n:
        return list(frames)
    scores = [sharpness(f) for f in frames]
    picked = []
    for chunk in np.array_split(np.arange(len(frames)), n):
        if len(chunk):
            picked.append(frames[int(chunk[int(np.argmax([scores[i] for i in chunk]))])])
    return picked


def extract_frames(data: bytes, n: int = 8, max_side: int = 640, max_read: int = 240,
                   suffix: str = ".webm") -> List[Image.Image]:
    """Вытаскивает до n чётких кадров из видео в байтах. Читает не больше max_read кадров (≈ 8 с при 30 fps)."""
    _need_cv2()
    fd, path = tempfile.mkstemp(suffix=suffix if suffix.startswith(".") else "." + suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise MediaError("Не удалось прочитать видео: формат не поддерживается.")
            frames = []
            while len(frames) < max_read:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(_shrink(frame, max_side))
        finally:
            cap.release()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if not frames:
        raise MediaError("В видео не нашлось ни одного кадра.")
    return [_to_pil(f) for f in pick_sharpest(frames, n)]


def parse_source(source: Union[str, int]) -> Union[str, int]:
    s = str(source).strip()
    return int(s) if s.isdigit() else s


def _is_live(src) -> bool:
    return isinstance(src, int) or str(src).lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "/dev/"))


def grab_from_camera(source: Union[str, int], n: int = 1, seconds: float = 0.0, warmup: int = 10,
                     max_side: int = 960, max_read: int = 240) -> List[Image.Image]:
    """Снимает с камеры устройства (индекс, /dev/videoN, rtsp://… или файл-имитация камеры для демо).

    n=1, seconds=0 — одно фото после прогрева (автоэкспозиция успевает отработать).
    seconds>0 — клип, из которого берутся n чётких кадров.
    """
    _need_cv2()
    src = parse_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise MediaError(f"Не удалось открыть камеру «{source}». Проверьте переменную WASHER_CAMERA.")
    try:
        live = _is_live(src)
        for _ in range(warmup if live else 0):
            cap.read()
        frames: List[np.ndarray] = []
        if seconds <= 0:
            ok, frame = cap.read()
            if not ok:
                raise MediaError("Камера не отдала кадр.")
            frames.append(_shrink(frame, max_side))
        else:
            t_end = time.time() + seconds
            while len(frames) < max_read and (not live or time.time() < t_end):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(_shrink(frame, max_side))
                if live:
                    time.sleep(0.04)                      # ≈ 25 кадров/с достаточно, не грузим процессор
        if not frames:
            raise MediaError("Камера не отдала ни одного кадра.")
        return [_to_pil(f) for f in pick_sharpest(frames, n)]
    finally:
        cap.release()


def sharpest_index(images: List[Image.Image]) -> int:
    """Индекс самого чёткого кадра (для превью результата)."""
    if cv2 is None or len(images) == 1:
        return 0
    scores = [sharpness(cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)) for im in images]
    return int(np.argmax(scores))


# ----------------------------------------------------------------------------------------------
# Загрузки: безопасное чтение картинки, превью, уборка
# ----------------------------------------------------------------------------------------------
def load_image_bytes(data: bytes, max_side: int = 1280) -> Image.Image:
    """Байты -> RGB-картинка с учётом поворота из EXIF (телефоны), уменьшенная до max_side."""
    import io
    from PIL import ImageOps, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(data))
        im.draft("RGB", (max_side * 2, max_side * 2))          # ускоряет декодирование больших JPEG
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise MediaError("Не удалось прочитать изображение: файл повреждён или это не картинка.")
    im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return im


def save_preview(im: Image.Image, directory, max_side: int = 800) -> str:
    """Сохраняет JPEG-превью под случайным именем (расширение задаём сами, имя от клиента не используем)."""
    import uuid
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    small = im.copy()
    small.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    name = f"{uuid.uuid4().hex}.jpg"
    small.convert("RGB").save(directory / name, "JPEG", quality=85)
    return name


def cleanup_old(directory, max_age_s: int = 3600) -> int:
    """Удаляет файлы старше max_age_s: фото белья — приватные данные, хранить их вечно незачем."""
    from pathlib import Path
    now, removed = time.time(), 0
    for p in Path(directory).glob("*"):
        try:
            if p.is_file() and now - p.stat().st_mtime > max_age_s:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed
