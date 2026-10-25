# backend/check_data.py
"""Проверка датасета ДО обучения: самое дешёвое место, где прячется потеря точности.

    python check_data.py ../TextileClass-7

Ищет:
  * разные написания одного класса ('Cotton_Mixed' / 'Cotton MIxed') и расхождение имён папок между train/valid/test;
  * перекос по классам;
  * размеры снимков (если почти всё огромное, при Resize до 224 пропадает фактура ткани);
  * дубликаты и почти-дубликаты МЕЖДУ split'ами (утечка делает valid/test оптимистичными, а реальную точность — неожиданной);
  * битые файлы.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from washing import normalize_name

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def dhash(path, size=8):
    """64-битный хэш по градиентам яркости: переживает ресайз/перекодирование JPEG."""
    with Image.open(path) as im:
        im.draft("L", (size * 8, size * 8))
        g = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def main(root: Path):
    splits = {}
    for name in ("train", "valid", "val", "test"):
        if (root / name).is_dir():
            splits[name] = root / name
    if not splits:
        raise SystemExit(f"В {root} нет train/valid/test")

    folders = {s: sorted(p.name for p in d.iterdir() if p.is_dir()) for s, d in splits.items()}
    print("Папки классов:")
    for s, f in folders.items():
        print(f"  {s:<6} {f}")
    canon_sets = {s: {normalize_name(x) for x in f} for s, f in folders.items()}
    ref = canon_sets.get("train") or next(iter(canon_sets.values()))
    for s, c in canon_sets.items():
        if c != ref:
            print(f"❌ {s}: канонические классы отличаются от train: {sorted(c ^ ref)}")
    for s, f in folders.items():
        groups = defaultdict(list)
        for x in f:
            groups[normalize_name(x)].append(x)
        for c, xs in groups.items():
            if len(xs) > 1:
                print(f"⚠️  {s}: папки {xs} — один класс '{c}' (train.py их объединит)")
    if len({tuple(f) for f in folders.values()}) > 1:
        print("⚠️  Имена папок различаются между split'ами: при наивном ImageFolder номера классов могут не совпасть.")

    records = []        # (split, class, path)
    bad = []
    sizes = []
    for s, d in splits.items():
        for cdir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in cdir.rglob("*"):
                if f.suffix.lower() in EXT:
                    records.append((s, normalize_name(cdir.name), f))
    print(f"\nВсего снимков: {len(records)}")
    table = defaultdict(lambda: defaultdict(int))
    for s, c, _ in records:
        table[c][s] += 1
    cols = list(splits)
    print(f"{'класс':<14}" + "".join(f"{c:>8}" for c in cols))
    for c in sorted(table):
        print(f"{c:<14}" + "".join(f"{table[c][s]:>8}" for s in cols))
    tr = [table[c]["train"] for c in table]
    if tr and min(tr) > 0 and max(tr) / min(tr) > 2.5:
        print(f"⚠️  Перекос классов в train: {max(tr) / min(tr):.1f}× (train.py включает веса классов по умолчанию)")

    hashes = []
    for s, c, f in records:
        try:
            with Image.open(f) as im:
                sizes.append(max(im.size))
            hashes.append(dhash(f))
        except Exception:                       # noqa: BLE001
            bad.append(f)
            hashes.append(0)
    if bad:
        print(f"❌ Не открылись {len(bad)} файлов, например {bad[:3]}")
    if sizes:
        q = np.percentile(sizes, [0, 50, 100]).astype(int)
        print(f"\nДлинная сторона снимков, px: min {q[0]} · медиана {q[1]} · max {q[2]}")
        if q[1] > 1000:
            print("ℹ️  Снимки крупные: Resize до 224 стирает фактуру ткани → в train.py вход 288 и случайные кропы исходного разрешения.")

    h = np.array(hashes, dtype=np.uint64)
    split_of = np.array([r[0] for r in records])
    cls_of = np.array([r[1] for r in records])
    leaks, within, label_clash = 0, 0, 0
    examples = []
    for i in range(len(records)):
        d = popcount(h[i + 1:] ^ h[i])
        for j in np.nonzero(d <= 4)[0] + i + 1:
            if split_of[i] != split_of[j]:
                leaks += 1
                if len(examples) < 5:
                    examples.append((records[i][2], records[j][2]))
            else:
                within += 1
            if cls_of[i] != cls_of[j]:
                label_clash += 1
    print(f"\nПочти-дубликаты между split'ами: {leaks} пар · внутри split'а: {within} пар")
    for a, b in examples:
        print(f"  {a}  ≈  {b}")
    if label_clash:
        print(f"❌ {label_clash} пар похожих снимков имеют РАЗНЫЕ метки — вероятная ошибка разметки.")
    if leaks:
        print("⚠️  Утечка train→valid/test завышает метрики. Пересоберите split так, чтобы дубликаты оказались по одну сторону.")
    if not (leaks or label_clash or bad):
        print("✅ Утечек, битых файлов и конфликтов меток не найдено.")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "../TextileClass-7"))
