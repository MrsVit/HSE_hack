# backend/washing.py
"""Логика выбора режима стирки. Без torch: её можно тестировать и запускать отдельно от модели.

Что делает:
  * profiles — программы по типам ткани (температура, отжим, время);
  * из вероятностей по областям кадра понимает, смешанная ли загрузка, и берёт самую нежную ткань
    (раньше UI писал «режим адаптирован под самую нежную ткань», но код брал top-1);
  * при низкой уверенности модели не гадает, а выбирает самый щадящий вариант из правдоподобных;
  * при разноцветной загрузке (colors.py) ограничивает температуру и время.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from colors import ColorReport


def normalize_name(name: str) -> str:
    """'Cotton_Mixed' / 'Cotton MIxed' / 'cotton mixed' -> 'Cotton Mixed'."""
    name = name.replace("_", " ").strip().replace("MIxed", "Mixed")
    return " ".join(w.capitalize() for w in name.split())


PROFILES: Dict[str, dict] = {
    "Cotton":       {"temp": 60, "spin": 1000, "time": 90, "name": "Хлопок",           "icon": "👕"},
    "Cotton Mixed": {"temp": 40, "spin": 800,  "time": 75, "name": "Смешанный хлопок", "icon": "👔"},
    "Denim":        {"temp": 30, "spin": 800,  "time": 75, "name": "Джинса",           "icon": "👖"},
    "Polyester":    {"temp": 40, "spin": 800,  "time": 60, "name": "Полиэстер",        "icon": "👚"},
    "Silk":         {"temp": 30, "spin": 400,  "time": 30, "name": "Шёлк",             "icon": "👘"},
    "Viscose":      {"temp": 30, "spin": 600,  "time": 45, "name": "Вискоза",          "icon": "👗"},
    "Wool":         {"temp": 30, "spin": 0,    "time": 45, "name": "Шерсть",           "icon": "🧶"},
}
FALLBACK_PROFILE = {"temp": 30, "spin": 400, "time": 45, "icon": "❓"}

# Ограничения при разноцветной загрузке. Краситель переходит в воду из-за температуры и времени,
# поэтому отжим цветом не ограничиваем.
COLOR_CAPS = {"high": {"temp": 30, "time": 45}, "medium": {"temp": 40, "time": 60}}


def profile_for(fabric: str) -> dict:
    base = PROFILES.get(fabric)
    if base is None:
        base = {**FALLBACK_PROFILE, "name": fabric}
    return dict(base)


def most_delicate(fabrics: Sequence[str]) -> str:
    """Самая нежная ткань = ниже температура, затем ниже отжим, затем короче программа."""
    return min(fabrics, key=lambda f: (profile_for(f)["temp"], profile_for(f)["spin"], profile_for(f)["time"]))


# ----------------------------------------------------------------------------------------------
# Вероятности по областям и кадрам
# ----------------------------------------------------------------------------------------------
def region_fabrics(region_probs: np.ndarray, class_names: Sequence[str],
                   min_conf: float = 0.55, min_frac: float = 0.3, min_regions: int = 2) -> List[Tuple[str, float]]:
    """Какие ткани реально присутствуют в кадре(ах).

    region_probs: (N кадров, R областей, C классов). Ткань «присутствует», если в её пользу
    уверенно (>= min_conf) проголосовало не меньше min_frac областей (и не меньше min_regions штук).
    Возвращает [(ткань, доля областей)], по убыванию доли.
    """
    flat = region_probs.reshape(-1, region_probs.shape[-1])
    total = len(flat)
    top = flat.argmax(1)
    conf = flat.max(1)
    out = []
    need = max(min_regions, math.ceil(min_frac * total))
    for c in range(flat.shape[1]):
        n = int(((top == c) & (conf >= min_conf)).sum())
        if n >= need:
            out.append((class_names[c], n / total))
    return sorted(out, key=lambda t: -t[1])


def apply_color_policy(profile: dict, report: Optional[ColorReport], enabled: bool = True) -> Tuple[dict, List[dict]]:
    """Смягчает режим при разноцветной загрузке. Возвращает (новый профиль, список изменений)."""
    profile = dict(profile)
    changes: List[dict] = []
    if report is None or report.severity == "ok" or not enabled:
        return profile, changes
    caps = COLOR_CAPS[report.severity]
    parts = []
    for key, unit, label in (("temp", "°C", "температура"), ("time", "мин", "время")):
        if profile[key] > caps[key]:
            parts.append(f"{label} {profile[key]} → {caps[key]} {unit}")
            changes.append({"param": key, "from": profile[key], "to": caps[key], "reason": "color_mix"})
            profile[key] = caps[key]
    if parts:
        profile["color_note"] = "Разноцветная загрузка: " + ", ".join(parts)
    return profile, changes


def _fabric_tips(fabric: str) -> List[str]:
    tips = {
        "Denim": ["🎨 Возможна линька. Стирайте отдельно или с похожими цветами.",
                  "💡 Выверните наизнанку для сохранения цвета."],
        "Silk": ["🧺 Рекомендуется использовать специальный мешок для стирки.",
                 "🚫 Не используйте отжим на высоких оборотах."],
        "Wool": ["💡 Используйте мешок для стирки, чтобы избежать усадки.",
                 "🧴 Не используйте обычный стиральный порошок (только для шерсти)."],
        "Viscose": ["💡 Стирайте при низкой температуре, ткань боится горячей воды."],
        "Polyester": ["✨ Быстро сохнет и почти не мнётся. Можно не гладить."],
        "Cotton": ["✨ Прочная натуральная ткань. Отлично переносит стандартный режим."],
        "Cotton Mixed": ["✨ Прочная натуральная ткань. Отлично переносит стандартный режим."],
    }
    return tips.get(fabric, [])


def build_response(class_names: Sequence[str], region_probs: np.ndarray,
                   color_report: Optional[ColorReport] = None, *, gentle_on_color_mix: bool = True,
                   conf_threshold: float = 0.5, frames_used: int = 1,
                   image_url: Optional[str] = None) -> dict:
    """Собирает ответ API из вероятностей по областям кадров и отчёта по цвету."""
    region_probs = np.asarray(region_probs, dtype=np.float64)
    if region_probs.ndim == 2:                       # (R, C) -> (1, R, C)
        region_probs = region_probs[None]
    probs = region_probs.mean((0, 1))
    order = np.argsort(-probs)
    top_name = class_names[int(order[0])]
    confidence = float(probs[order[0]])

    present = region_fabrics(region_probs, class_names)
    mixed = len(present) >= 2
    low_conf = confidence < conf_threshold
    adjustments: List[dict] = []

    if mixed:
        candidates = [n for n, _ in present]
    elif low_conf:
        candidates = [top_name] + [class_names[int(i)] for i in order[1:] if probs[int(i)] >= 0.2]
    else:
        candidates = [top_name]
    chosen = most_delicate(candidates)
    profile = profile_for(chosen)
    if chosen != top_name or len(candidates) > 1:
        why = "mixed_fabrics" if mixed else "low_confidence"
        profile["basis"] = ("Смешанная загрузка: берём самую нежную ткань — " if mixed
                            else "Модель не уверена: берём самый щадящий из правдоподобных вариантов — ") + profile["name"]
        if chosen != top_name:
            adjustments.append({"param": "fabric", "from": top_name, "to": chosen, "reason": why})

    profile, color_changes = apply_color_policy(profile, color_report, gentle_on_color_mix)
    adjustments += color_changes

    detected = []
    for i in order[:3]:
        conf = round(float(probs[int(i)]) * 100, 2)
        if conf > 1.0:
            detected.append({"fabric": class_names[int(i)], "confidence": conf, "profile": profile_for(class_names[int(i)])})

    tips: List[str] = []
    if low_conf and not mixed:
        tips.append("⚠️ Низкая уверенность модели — выбран самый щадящий из вероятных режимов. "
                    "Можно снять ещё кадр при лучшем освещении.")
    if mixed:
        tips.append("⚠️ Обнаружена смешанная загрузка: " + ", ".join(
            f"{profile_for(n)['name']} ({share:.0%} кадра)" for n, share in present) + ".")
    if color_report is not None and color_report.severity != "ok":
        tips.append(("🎨 " if color_report.severity == "medium" else "🚨 ") + color_report.message)
        tips.append("🧺 Самое надёжное — разделить: светлое отдельно, тёмное и яркое отдельно. "
                    "Если стирать вместе, добавьте салфетку-ловушку цвета.")
        if not gentle_on_color_mix:
            tips.append("ℹ️ Автоматическое смягчение режима по цвету выключено — режим выбран только по ткани.")
    for f in candidates:
        for t in _fabric_tips(f):
            if t not in tips:
                tips.append(t)
    if not tips:
        tips.append("✨ Стандартная стирка, всё под контролем!")

    return {
        "detected_fabrics": detected,
        "recommended_mode": profile,
        "confidence": round(confidence * 100, 2),
        "low_confidence": bool(low_conf),
        "mixed_load_warning": bool(mixed),
        "mixed_fabrics": [{"fabric": n, "share": round(s, 3)} for n, s in present],
        "color_check": color_report.to_dict() if color_report is not None else None,
        "adjustments": adjustments,
        "tips": tips,
        "frames_used": int(frames_used),
        **({"image_url": image_url} if image_url else {}),
    }
