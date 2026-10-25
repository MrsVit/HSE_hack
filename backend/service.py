# backend/service.py
"""Склейка: картинки -> классификатор + k-means по цвету -> ответ API.

Не зависит от FastAPI, поэтому её же используют тесты и можно вызвать из скрипта.
torch подгружается лениво: без обученной модели остальной код (цвет, видео, логика режимов) работает.
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

import colors
import washing

CONF_THRESHOLD = float(os.environ.get("FABRIC_CONF_THRESHOLD", "0.5"))


class ModelNotReady(RuntimeError):
    """Нет весов модели или не установлен torch."""


_classifier = None
_lock = threading.Lock()


def set_classifier(obj) -> None:
    """Подмена классификатора (тесты, заглушки). obj: .class_names и .predict(images, fast) -> (N, R, C)."""
    global _classifier
    _classifier = obj


def get_classifier():
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                try:
                    from model import FabricClassifier, WeightsMissing
                except ImportError as exc:
                    raise ModelNotReady(f"Не установлен torch/torchvision: {exc}") from exc
                try:
                    _classifier = FabricClassifier()
                except WeightsMissing as exc:
                    raise ModelNotReady(str(exc)) from exc
    return _classifier


def model_info() -> dict:
    try:
        clf = get_classifier()
        return {"loaded": True, "name": getattr(clf, "title", "fabric classifier"), "classes": list(clf.class_names)}
    except ModelNotReady as exc:
        return {"loaded": False, "error": str(exc)}


def analyze_images(images: List, *, gentle_on_color_mix: bool = True, fast: bool = False,
                   image_url: Optional[str] = None) -> dict:
    """images — список PIL-кадров (1 для фото, несколько для видео)."""
    clf = get_classifier()
    region_probs = clf.predict(images, fast=fast)
    report = colors.analyze_images(images)
    return washing.build_response(
        clf.class_names, region_probs, report,
        gentle_on_color_mix=gentle_on_color_mix, conf_threshold=CONF_THRESHOLD,
        frames_used=len(images), image_url=image_url,
    )
