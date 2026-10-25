"""Честная оценка чекпоинта: не только accuracy, но и то, что важно для стирки.
python evaluate.py                          # test, fabric_model_best.pth
python evaluate.py --ckpt old_model.pth     # старый чекпоинт (MobileNetV3-Small) читается тоже
python evaluate.py --split valid --no-tta
"""
import argparse
import json
import os
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import modelzoo as mz
from washing import PROFILES, normalize_name


# ============================================================
# ВСТРОЕННЫЕ ФУНКЦИИ (вместо импорта из train.py)
# ============================================================
def find_split(root: Path, name: str, fallback: str = "val") -> Path:
    """Найти папку split: если name не существует, пробуем fallback."""
    p = root / name
    if p.is_dir():
        return p
    if fallback and (root / fallback).is_dir():
        return root / fallback
    raise FileNotFoundError(f"В {root} нет папок {name} или {fallback}")


EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class CanonFolder(torch.utils.data.Dataset):
    """Dataset с каноническими (нормализованными) именами классов."""

    def __init__(self, root, canon, loader=None, transform=None):
        self.root = Path(root)
        self.canon = list(canon)
        self.loader = loader or (lambda p: Image.open(p).convert("RGB"))
        self.transform = transform
        self.samples = []
        for cdir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            canon_name = normalize_name(cdir.name)
            if canon_name not in self.canon:
                continue
            label = self.canon.index(canon_name)
            for f in cdir.rglob("*"):
                if f.suffix.lower() in EXT:
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# ОСНОВНОЙ КОД
# ============================================================
def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    names = [normalize_name(n) for n in ckpt["class_names"]]
    if "arch" in ckpt:
        model = mz.build_model(ckpt["arch"], len(names))
        img, temp, eval_tf = int(ckpt.get("img_size", 288)), float(ckpt.get("temperature", 1.0)), None
    else:                                            # старый train.py
        model = models.mobilenet_v3_small()
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[0].in_features, len(names)))
        img, temp = 224, 1.0
        eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mz.MEAN, mz.STD)])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), names, img, temp, eval_tf, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "../TextileClass-7"))
    ap.add_argument("--ckpt", default="fabric_model_best.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--out", default="evaluation_report.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, img, temp, legacy_tf, ckpt = load_checkpoint(args.ckpt, device)
    k = len(names)

    split_dir = find_split(Path(args.data), args.split, "val") if args.split == "valid" else find_split(Path(args.data), args.split)
    loader_fn = partial(mz.load_image, max_side=640)
    dl = dict(num_workers=args.workers, pin_memory=device.type == "cuda")

    use_tta = not args.no_tta and legacy_tf is None
    tf = mz.MultiView(img) if use_tta else (legacy_tf or mz.eval_transform(img))
    ds = CanonFolder(split_dir, canon=names, loader=loader_fn, transform=tf)
    loader = DataLoader(ds, batch_size=max(1, args.batch // 4) if use_tta else args.batch, shuffle=False, **dl)

    if use_tta:
        probs, y = mz.collect_probs_tta(model, loader, device, temp)
    else:
        logits, y = mz.collect_logits(model, loader, device)
        probs = torch.softmax(logits / temp, dim=1)

    probs, y = probs.numpy(), y.numpy()
    pred, conf = probs.argmax(1), probs.max(1)
    n = len(y)
    cm = mz.confusion_matrix(y, pred, k)
    prec, rec, f1, macro = mz.per_class_report(cm)
    acc = np.trace(cm) / n

    print(f"\n{args.ckpt} · {args.split} · {n} снимков · {'мульти-кроп' if use_tta else 'один кроп'} · T={temp:.2f}")
    print(f"Accuracy {100 * acc:.2f}% · macro-F1 {100 * macro:.1f}%\n")
    print(f"{'класс':<14}{'снимков':>8}{'precision':>11}{'recall':>8}{'F1':>7}")
    for i, name in enumerate(names):
        print(f"{name:<14}{cm[i].sum():>8}{100 * prec[i]:>10.1f}%{100 * rec[i]:>7.1f}%{100 * f1[i]:>6.1f}%")

    off = [(cm[i, j], names[i], names[j]) for i in range(k) for j in range(k) if i != j and cm[i, j]]
    print("\nЧаще всего путает (правда → ответ):")
    for c, a, b in sorted(off, reverse=True)[:6]:
        print(f"  {a} → {b}: {c}")

    # --- Практические метрики: сравниваем программы стирки, а не названия классов ---
    progs = [(PROFILES[n]["temp"], PROFILES[n]["spin"]) if n in PROFILES else (0, 0) for n in names]
    same_prog = np.mean([progs[a] == progs[b] for a, b in zip(y, pred)])
    harsher = np.mean([progs[b][0] > progs[a][0] or progs[b][1] > progs[a][1] for a, b in zip(y, pred)])
    print(f"\nПрактическая точность (та же температура и отжим): {100 * same_prog:.1f}%")
    print(f"Опасные ошибки (режим жёстче, чем нужно ткани):   {100 * harsher:.1f}% снимков")

    print("\nПорог уверенности → покрытие / точность на ответах:")
    sel = []
    for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= th
        cov, a = m.mean(), (pred[m] == y[m]).mean() if m.any() else float("nan")
        sel.append({"threshold": th, "coverage": float(cov), "accuracy": float(a)})
        print(f"  ≥ {th:.1f}: отвечает на {100 * cov:5.1f}% снимков, прав в {100 * a:5.1f}%")

    Path(args.out).write_text(json.dumps({
        "ckpt": args.ckpt, "split": args.split, "n": n, "tta": use_tta, "temperature": temp,
        "accuracy": float(acc), "macro_f1": macro, "classes": names, "confusion": cm.tolist(),
        "per_class": {nm: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])} for i, nm in enumerate(names)},
        "practical_accuracy": float(same_prog), "harsher_than_needed": float(harsher), "selective": sel,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {args.out}")


if __name__ == "__main__":
    main()