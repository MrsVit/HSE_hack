# backend/colors.py
"""Проверка «разноцветности» загрузки методом k-means.

Зачем: если в барабане рядом лежат белая и красная/чёрная вещи, один режим подходит плохо:
краситель переходит на светлое. Модуль находит в кадре (или в нескольких кадрах видео) крупные
пятна заметно разного цвета и говорит, насколько это опасно: ok / medium / high.

Как работает:
  1. Кадр уменьшается до ~96 px, пиксели переводятся в CIE Lab (там евклидово расстояние ≈
     воспринимаемая разница цветов, в RGB это не так).
  2. k-means (k-means++, k=5) режет пиксели на цветовые группы; близкие группы (ΔE < merge_de)
     склеиваются, так что складки и тени одной вещи остаются одной группой.
  3. Для каждой группы считается самая большая СВЯЗНАЯ область. Вещь — это компактное пятно, а
     полоски, принт и шум дают россыпь мелких кусочков и отсеиваются (полосатая футболка — одна вещь).
  4. Группы из разных кадров сводятся в общую палитру: барабан вращается, и в разных кадрах видны
     разные вещи.
  5. Правила стирки (светлое + тёмное/яркое — опасно, два разных ярких цвета — осторожно).

Зависимости: только numpy и Pillow (оба уже нужны проекту).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------------------------
# Настройки. Все пороги в одном месте, чтобы их было удобно подкрутить под реальную камеру.
# ----------------------------------------------------------------------------------------------


@dataclass
class ColorConfig:
    work_size: int = 96          # длинная сторона рабочей копии кадра
    k: int = 5                   # сколько цветовых групп ищет k-means до склейки
    max_samples: int = 3000      # сколько пикселей отдаём в k-means (остальные присваиваются по центрам)
    merge_de: float = 20.0       # группы ближе этого ΔE (по Lab) считаем одним цветом
    min_share: float = 0.08      # минимальная площадь компактного пятна, чтобы считать его отдельной вещью
    min_L: float = 2.0           # пиксели темнее — чёрные поля кадра/виньетка, игнорируем
    seed: int = 7


# Границы категорий в Lab. L — светлота 0..100, C — насыщенность (chroma).
LIGHT_L, LIGHT_C = 72.0, 25.0    # светлое: белый, молочный, бледно-пастельный
DARK_L, DARK_C = 32.0, 35.0      # тёмное: чёрный, графит, тёмно-синий
VIVID_C = 35.0                   # яркое: насыщенный цвет, оттенок важен
GREY_C = 12.0                    # серое: почти без цвета и не светлое/не тёмное
HUE_GAP = 45.0                   # разница оттенков, после которой цвета считаются разными

SEVERITY_ORDER = {"ok": 0, "medium": 1, "high": 2}


# ----------------------------------------------------------------------------------------------
# Цвет
# ----------------------------------------------------------------------------------------------
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8 или float 0..1), форма (..., 3) -> CIE Lab (L 0..100, a и b примерно ±128)."""
    x = np.asarray(rgb, dtype=np.float64)
    if x.max() > 1.0:
        x = x / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _hue_deg(a: float, b: float) -> float:
    return float(np.degrees(np.arctan2(b, a)) % 360.0)


def _hue_diff(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def delta_e(l1: np.ndarray, l2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(l1) - np.asarray(l2)))


# ----------------------------------------------------------------------------------------------
# k-means (k-means++), чистый numpy
# ----------------------------------------------------------------------------------------------
def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30, n_init: int = 3):
    """x: (n, d). Возвращает (центры (k, d), метки (n,)). Лучший из n_init запусков по инерции."""
    n = len(x)
    k = max(1, min(k, n))
    best = None
    for _ in range(n_init):
        centers = [x[rng.integers(n)]]
        d2 = ((x - centers[0]) ** 2).sum(1)
        for _ in range(1, k):
            tot = d2.sum()
            if tot <= 1e-12:                       # все точки уже совпали с центрами
                centers.append(x[rng.integers(n)])
            else:
                centers.append(x[rng.choice(n, p=d2 / tot)])
            d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
        c = np.array(centers, dtype=np.float64)
        for _ in range(iters):
            dist = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            lab = dist.argmin(1)
            new = c.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    new[j] = x[m].mean(0)
            moved = np.abs(new - c).max()
            c = new
            if moved < 1e-3:
                break
        dist = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = dist.argmin(1)
        inertia = float(dist[np.arange(n), lab].sum())
        if best is None or inertia < best[0]:
            best = (inertia, c, lab)
    return best[1], best[2]


def _merge_close(centers: np.ndarray, weights: np.ndarray, thr: float):
    """Склеивает ближайшие центры, пока расстояние между ними < thr. Возвращает отображение старый->новый."""
    groups = [[i] for i in range(len(centers))]
    cen = [centers[i].copy() for i in range(len(centers))]
    wt = [float(weights[i]) + 1e-9 for i in range(len(centers))]
    while len(groups) > 1:
        best, pair = None, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = float(np.linalg.norm(cen[i] - cen[j]))
                if best is None or d < best:
                    best, pair = d, (i, j)
        if best >= thr:
            break
        i, j = pair
        cen[i] = (cen[i] * wt[i] + cen[j] * wt[j]) / (wt[i] + wt[j])
        wt[i] += wt[j]
        groups[i] += groups[j]
        del groups[j], cen[j], wt[j]
    mapping = np.zeros(len(centers), dtype=int)
    for new_id, g in enumerate(groups):
        for old in g:
            mapping[old] = new_id
    return mapping, np.array(cen)


def _largest_component(mask: np.ndarray) -> int:
    """Размер самой большой 4-связной области в булевой маске."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        size = 0
        dq = deque([(y0, x0)])
        seen[y0, x0] = True
        while dq:
            y, x = dq.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    dq.append((ny, nx))
        best = max(best, size)
    return best


# ----------------------------------------------------------------------------------------------
# Описание цвета
# ----------------------------------------------------------------------------------------------
@dataclass
class ColorGroup:
    hex: str
    name: str
    category: str      # light | dark | grey | vivid | muted
    share: float       # какую часть кадра занимает в среднем по кадрам (только компактные пятна)
    peak_share: float  # максимум по кадрам
    L: float
    C: float
    h: float
    rgb: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("L", "C", "h"):
            d[k] = round(d[k], 1)
        d["share"] = round(d["share"], 3)
        d["peak_share"] = round(d["peak_share"], 3)
        return d


def categorize(L: float, C: float) -> str:
    if C >= VIVID_C:
        return "vivid"
    if L >= LIGHT_L and C < LIGHT_C:
        return "light"
    if L <= DARK_L:
        return "dark"
    if C < GREY_C:
        return "grey"
    return "muted"


def color_name(L: float, C: float, h: float) -> str:
    cat = categorize(L, C)
    if cat == "light":
        return "белый" if C < 8 else "светлый (молочный/пастельный)"
    if cat == "dark":
        return "чёрный" if (L < 18 and C < 12) else "тёмный"
    if cat == "grey":
        return "серый"
    if h >= 350 or h < 55:
        base = "красный"
        if L > 68:
            base = "розовый"
        return base
    if h < 85:
        return "коричневый" if L < 55 else "оранжевый"
    if h < 120:
        return "жёлтый"
    if h < 180:
        return "зелёный"
    if h < 235:
        return "бирюзовый"
    if h < 315:
        return "синий"
    return "фиолетовый"


def _hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(round(max(0, min(255, v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


# ----------------------------------------------------------------------------------------------
# Отчёт
# ----------------------------------------------------------------------------------------------
@dataclass
class Conflict:
    a: str
    b: str
    level: str      # medium | high
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ColorReport:
    severity: str                 # ok | medium | high
    score: float                  # 0..1, для индикатора
    message: str
    palette: List[ColorGroup] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    n_frames: int = 1

    @property
    def mixed(self) -> bool:
        return self.severity != "ok"

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "mixed": self.mixed,
            "score": round(self.score, 2),
            "message": self.message,
            "palette": [g.to_dict() for g in self.palette],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "n_frames": self.n_frames,
        }


def _pair_level(a: ColorGroup, b: ColorGroup) -> Optional[Conflict]:
    """Правила сортировки белья. None — пара безопасна."""
    for light, other in ((a, b), (b, a)):
        if light.category != "light":
            continue
        if other.category in ("vivid", "dark"):
            return Conflict(light.name, other.name, "high",
                            f"{other.name} краситель может окрасить {light.name} вещи")
        if other.category == "muted" and other.L < 55:
            return Conflict(light.name, other.name, "medium",
                            f"{other.name} цвет может полинять на {light.name}")
    if a.category == "vivid" and b.category == "vivid" and _hue_diff(a.h, b.h) >= HUE_GAP:
        return Conflict(a.name, b.name, "medium", f"два разных ярких цвета ({a.name}, {b.name}) могут линять друг на друга")
    for vivid, other in ((a, b), (b, a)):
        if vivid.category == "vivid" and other.category == "dark":
            return Conflict(vivid.name, other.name, "medium", f"яркое ({vivid.name}) и тёмное ({other.name}) лучше стирать раздельно")
    return None


def _message(severity: str, conflicts: List[Conflict], palette: List[ColorGroup]) -> str:
    if severity == "ok":
        if not palette:
            return "Крупных однотонных областей не нашлось (узор или мелкие детали) — по цвету разделить не удалось."
        if len(palette) <= 1:
            return "Загрузка однородная по цвету — отдельная сортировка не нужна."
        return "Цвета вещей близки или безопасно сочетаются — отдельная сортировка не нужна."
    top = conflicts[0]
    if severity == "high":
        return (f"В кадре сильно различаются по цвету вещи: {top.a} и {top.b}. "
                "Риск, что краситель окрасит светлое. Лучше разделить загрузку; "
                "если стирать вместе — режим смягчён.")
    return (f"В кадре заметно разные по цвету вещи: {top.a} и {top.b}. "
            "Возможна линька, режим смягчён.")


# ----------------------------------------------------------------------------------------------
# Анализ
# ----------------------------------------------------------------------------------------------
def _to_work_array(img: Image.Image, size: int) -> np.ndarray:
    im = img.convert("RGB")
    im.thumbnail((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _frame_groups(arr: np.ndarray, cfg: ColorConfig, rng: np.random.Generator):
    """k-means по одному кадру -> список компактных цветовых пятен [(Lab-центр, RGB, доля пятна)]."""
    h, w, _ = arr.shape
    lab = rgb_to_lab(arr).reshape(-1, 3)
    rgb = arr.reshape(-1, 3).astype(np.float64)
    valid = lab[:, 0] >= cfg.min_L
    if valid.sum() < 50:
        return []
    idx = np.nonzero(valid)[0]
    sample = idx if len(idx) <= cfg.max_samples else rng.choice(idx, cfg.max_samples, replace=False)
    centers, _ = kmeans(lab[sample], cfg.k, rng)

    dist = ((lab[idx][:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    lab_id = dist.argmin(1)
    weights = np.bincount(lab_id, minlength=len(centers)) / len(idx)
    mapping, merged = _merge_close(centers, weights, cfg.merge_de)
    lab_id = mapping[lab_id]

    label_map = np.full(h * w, -1, dtype=int)
    label_map[idx] = lab_id
    label_map = label_map.reshape(h, w)

    out = []
    for g in range(len(merged)):
        mask = label_map == g
        n = int(mask.sum())
        if n == 0:
            continue
        blob = _largest_component(mask) / len(idx)    # доля кадра, занятая самым крупным куском
        if blob < cfg.min_share:
            continue                                  # россыпь (полоски, принт, шум) — не отдельная вещь
        mean_lab = lab[mask.reshape(-1)].mean(0)
        mean_rgb = rgb[mask.reshape(-1)].mean(0)
        out.append((mean_lab, mean_rgb, blob))
    return out


def analyze_images(images: Iterable[Image.Image], cfg: Optional[ColorConfig] = None) -> ColorReport:
    """images — один или несколько кадров (PIL). Возвращает ColorReport."""
    cfg = cfg or ColorConfig()
    rng = np.random.default_rng(cfg.seed)
    frames = list(images)
    n_frames = max(1, len(frames))

    pooled = []            # (lab, rgb, blob_share)
    for im in frames:
        pooled += _frame_groups(_to_work_array(im, cfg.work_size), cfg, rng)

    # Сводим пятна из разных кадров в общую палитру.
    palette_groups: List[dict] = []
    for lab_c, rgb_c, blob in sorted(pooled, key=lambda t: -t[2]):
        for g in palette_groups:
            if delta_e(g["lab"], lab_c) < cfg.merge_de:
                tot = g["w"] + blob
                g["lab"] = (g["lab"] * g["w"] + lab_c * blob) / tot
                g["rgb"] = (g["rgb"] * g["w"] + rgb_c * blob) / tot
                g["w"] = tot
                g["shares"].append(blob)
                break
        else:
            palette_groups.append({"lab": lab_c.copy(), "rgb": rgb_c.copy(), "w": blob, "shares": [blob]})

    palette: List[ColorGroup] = []
    for g in palette_groups:
        L, a, b = g["lab"]
        C, h = float(np.hypot(a, b)), _hue_deg(a, b)
        palette.append(ColorGroup(
            hex=_hex(g["rgb"]), name=color_name(L, C, h), category=categorize(L, C),
            share=float(sum(g["shares"]) / n_frames), peak_share=float(max(g["shares"])),
            L=float(L), C=C, h=h, rgb=[int(round(v)) for v in g["rgb"]],
        ))
    palette.sort(key=lambda g: -g.share)

    # Оцениваем пары. Одинаковые названия (например, два разных "тёмных") в сообщении объединяем.
    conflicts: List[Conflict] = []
    seen = set()
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            c = _pair_level(palette[i], palette[j])
            if c and (c.a, c.b, c.level) not in seen:
                seen.add((c.a, c.b, c.level))
                conflicts.append(c)
    conflicts.sort(key=lambda c: -SEVERITY_ORDER[c.level])

    severity = conflicts[0].level if conflicts else "ok"
    spread = 0.0
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            spread = max(spread, delta_e(_lab_of(palette[i]), _lab_of(palette[j])))
    score = {"ok": 0.0, "medium": 0.5, "high": 0.75}[severity] + (0.25 if severity != "ok" else 0.3) * min(1.0, spread / 100.0)
    return ColorReport(severity=severity, score=min(1.0, score), message=_message(severity, conflicts, palette),
                       palette=palette, conflicts=conflicts, n_frames=n_frames)


def _lab_of(g: ColorGroup) -> np.ndarray:
    return np.array([g.L, g.C * np.cos(np.radians(g.h)), g.C * np.sin(np.radians(g.h))])
