# backend/model.py
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, List


def _normalize_name(name: str) -> str:
    """'Cotton_Mixed' / 'Cotton MIxed' / 'cotton mixed' -> 'Cotton Mixed'"""
    name = name.replace("_", " ").strip()
    name = name.replace("MIxed", "Mixed")
    return " ".join(w.capitalize() for w in name.split())


class FabricClassifier:
    def __init__(self):
        print("🧠 Загружаю MobileNetV3-Large (TextileClass-7)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "fabric_model_best.pth")
        classes_path = os.path.join(base_dir, "class_names.txt")

        # 1. Имена классов
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                raw_names = [line.strip() for line in f if line.strip()]
        else:
            raw_names = [
                "Cotton", "Cotton Mixed", "Denim",
                "Polyester", "Silk", "Viscose", "Wool",
            ]
        self.class_names = [_normalize_name(n) for n in raw_names]

        # 2. Загружаем checkpoint ДО создания модели — чтобы узнать arch
        checkpoint = None
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            ckpt_classes = checkpoint.get("class_names", raw_names)
            self.class_names = [_normalize_name(n) for n in ckpt_classes]
            arch = checkpoint.get("arch", "mobilenet_v3_large")  # дефолт — Large
        else:
            print("⚠️  Файл fabric_model_best.pth не найден! Сначала запусти train.py")
            arch = "mobilenet_v3_large"

        # 3. Архитектура — Large (совпадает с train.py)
        if arch == "mobilenet_v3_large":
            self.model = models.mobilenet_v3_large()
            num_features = self.model.classifier[0].in_features   # 960
            print("   Архитектура: MobileNetV3-Large")
        else:
            # fallback на Small, если вдруг старый checkpoint
            self.model = models.mobilenet_v3_small()
            num_features = self.model.classifier[0].in_features   # 576
            print("   Архитектура: MobileNetV3-Small (fallback)")

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, len(self.class_names)),
        )

        # 4. Веса
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            val_acc = checkpoint.get("val_acc", 0)
            print(f"✅ Веса загружены! (Val Acc: {val_acc:.2f}%)")

        self.model.to(self.device)
        self.model.eval()

        # 5. Трансформации — как в train.py (val)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 6. Профили стирки
        self.washing_profiles = {
            "Cotton":       {"temp": 60, "spin": 1000, "time": 90, "name": "Хлопок",           "icon": "👕"},
            "Cotton Mixed": {"temp": 40, "spin": 800,  "time": 75, "name": "Смешанный хлопок", "icon": "👔"},
            "Denim":        {"temp": 30, "spin": 800,  "time": 75, "name": "Джинса",           "icon": "👖"},
            "Polyester":    {"temp": 40, "spin": 800,  "time": 60, "name": "Полиэстер",        "icon": "👚"},
            "Silk":         {"temp": 30, "spin": 400,  "time": 30, "name": "Шёлк",             "icon": "👘"},
            "Viscose":      {"temp": 30, "spin": 600,  "time": 45, "name": "Вискоза",          "icon": "👗"},
            "Wool":         {"temp": 30, "spin": 0,    "time": 45, "name": "Шерсть",           "icon": "🧶"},
        }

        print("✅ Модель готова к анализу!")

    def analyze(self, image_path: str) -> Dict:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probs, 0)

        fabric_name = self.class_names[predicted_idx.item()]
        conf_percent = round(float(confidence) * 100, 2)

        all_probs = []
        for i, prob in enumerate(probs):
            conf = round(float(prob) * 100, 2)
            if conf > 1.0:
                cls_name = self.class_names[i]
                profile = self.washing_profiles.get(cls_name, {
                    "temp": 30, "spin": 400, "time": 45,
                    "name": cls_name, "icon": "❓",
                })
                all_probs.append({
                    "fabric": cls_name,
                    "confidence": conf,
                    "profile": profile,
                })
        all_probs.sort(key=lambda x: x["confidence"], reverse=True)

        profile = self.washing_profiles.get(fabric_name, {
            "temp": 30, "spin": 400, "time": 45,
            "name": fabric_name, "icon": "❓",
        })

        mixed_load = len(all_probs) > 1 and all_probs[1]["confidence"] > 15

        return {
            "detected_fabrics": all_probs[:3],
            "recommended_mode": profile,
            "mixed_load_warning": mixed_load,
            "tips": self._generate_tips(fabric_name, mixed_load, conf_percent),
        }

    def _generate_tips(self, fabric: str, mixed_load: bool, confidence: float) -> List[str]:
        tips = []
        if confidence < 50:
            tips.append("⚠️ Низкая уверенность модели. Рекомендуется безопасный режим (30°C).")
        if mixed_load:
            tips.append("⚠️ Обнаружена смешанная загрузка. Режим адаптирован под самую нежную ткань.")

        if fabric == "Denim":
            tips.append("🎨 Возможна линька. Стирайте отдельно или с похожими цветами.")
            tips.append("💡 Выверните наизнанку для сохранения цвета.")
        elif fabric == "Silk":
            tips.append("🧺 Рекомендуется использовать специальный мешок для стирки.")
            tips.append("🚫 Не используйте отжим на высоких оборотах.")
        elif fabric == "Wool":
            tips.append("💡 Используйте мешок для стирки, чтобы избежать усадки.")
            tips.append("🧴 Не используйте обычный стиральный порошок (только для шерсти).")
        elif fabric == "Viscose":
            tips.append("💡 Стирайте при низкой температуре, ткань боится горячей воды.")
        elif fabric == "Polyester":
            tips.append("✨ Быстро сохнет и почти не мнётся. Можно не гладить.")
        elif fabric in ("Cotton", "Cotton Mixed"):
            tips.append("✨ Прочная натуральная ткань. Отлично переносит стандартный режим.")

        if not tips:
            tips.append("✨ Стандартная стирка, всё под контролем!")
        return tips


_classifier = None


def get_classifier() -> FabricClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FabricClassifier()
    return _classifier