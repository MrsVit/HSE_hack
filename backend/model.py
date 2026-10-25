# backend/model.py
import os
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

# Импорт того, что нужно
from washing import normalize_name, profile_for

class WeightsMissing(RuntimeError):
    pass

class FabricClassifier:
    def __init__(self):
        print("🧠 Загружаю классификатор ткани...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "fabric_model_best.pth")
        
        if not os.path.exists(self.model_path):
            raise WeightsMissing(f"Нет файла весов {self.model_path}. Запусти train.py")
            
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
        
        # Берем классы из чекпоинта
        ckpt_classes = ckpt.get("class_names", [])
        self.class_names = [normalize_name(c) for c in ckpt_classes]
        self.title = f"mobilenet_v3_large (TextileClass-{len(self.class_names)})"
        
        # --- ТОЧНО КАК В TRAIN.PY ---
        self.model = models.mobilenet_v3_large(weights=None) # без предобучения, т.к. загрузим свои
        num_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, len(self.class_names)),
        )
        # -----------------------------
        
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
        except Exception as e:
            raise WeightsMissing(f"Ошибка загрузки весов: {e}")
            
        self.model.to(self.device).eval()
        self.img_size = 224 # В train.py используется 224
        self.temperature = 1.0

    @torch.no_grad()
    def predict(self, images: List[Image.Image], fast: bool = False) -> np.ndarray:
        from torchvision import transforms
        # Простой трансформ, как в train.py
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        batch = torch.stack([tf(img.convert("RGB")) for img in images]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Возвращаем (N_frames, 1, C_classes) - 1 "регион" (весь кадр)
        return probs[:, None, :]

    def analyze(self, images: List[Image.Image]) -> Dict:
        # Оставил на случай, если main.py вызывает напрямую
        probs = self.predict(images)
        mean = probs.mean((0, 1))
        order = np.argsort(-mean)
        top_name = self.class_names[int(order[0])]
        conf = round(float(mean[order[0]]) * 100, 2)
        
        return {
            "detected_fabrics": [{"fabric": top_name, "confidence": conf, "profile": profile_for(top_name)}],
            "recommended_mode": profile_for(top_name),
            "confidence": conf,
            "mixed_load_warning": False,
            "tips": ["✨ Стандартная стирка."]
        }

_classifier = None
def get_classifier() -> FabricClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FabricClassifier()
    return _classifier