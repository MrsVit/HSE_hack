# backend/modelzoo.py
"""Общее для train.py, evaluate.py и model.py: архитектуры, трансформации, мульти-кроп инференс.

Единый источник правды важен: раньше архитектура и препроцессинг дублировались в train.py и model.py
«на глаз», и любое расхождение молча давало мусор на инференсе.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Все есть в torchvision, ничего лишнего ставить не нужно.
ARCHS = ("convnext_tiny", "efficientnet_v2_s", "resnet50", "mobilenet_v3_large", "mobilenet_v3_small")
N_REGIONS = 5          # 4 угла + центр
N_VIEWS = N_REGIONS * 2  # каждая область ещё и в зеркальном виде


# ----------------------------------------------------------------------------------------------
# Модель
# ----------------------------------------------------------------------------------------------
def _replace_last_linear(model: nn.Module, num_classes: int, dropout: float) -> str:
    """Заменяет последний nn.Linear на Dropout+Linear(num_classes). Работает для всех ARCHS. Возвращает его путь."""
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last = name
    if last is None:
        raise ValueError("В модели нет nn.Linear")
    parent_name, _, child = last.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    old = getattr(parent, child)
    setattr(parent, child, nn.Sequential(nn.Dropout(dropout), nn.Linear(old.in_features, num_classes)))
    return last


def build_model(arch: str, num_classes: int, pretrained: bool = False, dropout: float = 0.3) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Неизвестная архитектура {arch!r}. Доступно: {', '.join(ARCHS)}")
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)
    model.head_path = _replace_last_linear(model, num_classes, dropout)      # строка, не Module
    return model


def head_parameters(model: nn.Module) -> List[nn.Parameter]:
    return list(model.get_submodule(model.head_path).parameters())


# ----------------------------------------------------------------------------------------------
# Загрузка картинок
# ----------------------------------------------------------------------------------------------
def load_image(path, max_side: int = 640) -> Image.Image:
    """RGB с учётом EXIF-поворота. Большие снимки сразу уменьшаем: декодировать 12 Мп на каждой эпохе дорого,
    а для текстуры хватает ~640 px по длинной стороне."""
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return im


# ----------------------------------------------------------------------------------------------
# Трансформации
# ----------------------------------------------------------------------------------------------
def train_transform(img_size: int) -> T.Compose:
    """Случайные кропы с исходным разрешением (сохраняют фактуру ткани) + TrivialAugment.
    Раньше: Resize((224, 224)) всего кадра — это сплющивало пропорции и размывало переплетение нитей."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.3, 1.0), ratio=(0.75, 1.333), interpolation=IM.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(0.3),
        T.TrivialAugmentWide(interpolation=IM.BILINEAR),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), value="random"),
    ])


def eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(round(img_size * 1.14)), interpolation=IM.BICUBIC),   # короткая сторона, пропорции сохраняются
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class MultiView:
    """Кадр -> 10 видов (N_VIEWS, 3, S, S): 5 областей (углы+центр) × (оригинал, зеркало).
    Порядок: [область0, область0-зеркало, область1, область1-зеркало, …]."""

    def __init__(self, img_size: int, scale: float = 1.3):
        self.resize = T.Resize(int(round(img_size * scale)), interpolation=IM.BICUBIC)
        self.five = T.FiveCrop(img_size)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        views = []
        for crop in self.five(self.resize(img)):
            views.append(self.norm(crop))
            views.append(self.norm(TF.hflip(crop)))
        return torch.stack(views)


# ----------------------------------------------------------------------------------------------
# Инференс
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_region_probs(model: nn.Module, images: Sequence[Image.Image], img_size: int, device,
                         temperature: float = 1.0, fast: bool = False, chunk: int = 16) -> np.ndarray:
    """Вероятности по областям кадров: форма (N, R, C). R=5 в обычном режиме, R=1 при fast=True.

    Зеркальные копии усредняются внутри области, а области остаются отдельно: по ним washing.py
    понимает, что в кадре лежат разные ткани. Среднее по областям — предсказание для кадра."""
    model.eval()
    if fast:
        tf = eval_transform(img_size)
        batch = torch.stack([tf(im) for im in images]).unsqueeze(1)                 # (N, 1, 3, S, S)
    else:
        mv = MultiView(img_size)
        batch = torch.stack([mv(im) for im in images])                              # (N, 10, 3, S, S)
    n, v = batch.shape[:2]
    flat = batch.reshape(n * v, *batch.shape[2:])
    outs = []
    for i in range(0, len(flat), chunk):
        logits = model(flat[i:i + chunk].to(device)).float() / max(temperature, 1e-3)
        outs.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(outs).reshape(n, v, -1)
    if not fast:
        probs = probs.reshape(n, N_REGIONS, 2, -1).mean(2)                          # усредняем зеркало
    return probs.numpy()


# ----------------------------------------------------------------------------------------------
# Оценка по DataLoader (общее для train.py и evaluate.py)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model: nn.Module, loader, device):
    """Один вид на снимок (центральный кроп). Возвращает (logits (N, C), labels (N,)) на CPU."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def collect_probs_tta(model: nn.Module, loader, device, temperature: float = 1.0):
    """Датасет должен отдавать MultiView -> (B, 10, 3, S, S). Возвращает (probs (N, C), labels (N,))."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        b, v = x.shape[:2]
        logits = model(x.reshape(b * v, *x.shape[2:]).to(device)).float() / max(temperature, 1e-3)
        probs.append(torch.softmax(logits, dim=1).reshape(b, v, -1).mean(1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def confusion_matrix(y_true, y_pred, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m


def per_class_report(m: np.ndarray):
    """precision, recall, f1 по классам (0, если делить не на что) и macro-F1."""
    tp = np.diag(m).astype(float)
    prec = np.divide(tp, m.sum(0), out=np.zeros_like(tp), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros_like(tp), where=m.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, float(f1.mean())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Подбирает T, при котором softmax(logits/T) лучше всего откалиброван на валидации.
    Без калибровки порог «уверенность ≥ 50%» ничего не значит: сети обычно самоуверенны."""
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / t.clamp_min(0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp_min(0.05))
