# backend/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Large_Weights
from torch.utils.data import DataLoader


def normalize_name(name: str) -> str:
    name = name.replace("_", " ").strip()
    name = name.replace("MIxed", "Mixed")
    return " ".join(w.capitalize() for w in name.split())


def main():
    print("🚀 Начинаю обучение (MobileNetV3-Large, fine-tune)...")

    DATA_DIR = os.environ.get("DATA_DIR", "../TextileClass-7")
    BATCH_SIZE = 32
    EPOCHS = 15
    LR_HEAD = 1e-3       # для нового классификатора
    LR_BACKBONE = 1e-4   # для размороженных слоёв базы

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(DATA_DIR):
        print(f"❌ Папка {DATA_DIR} не найдена!")
        return

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=transform_val)
    test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=transform_val)

    class_names = [normalize_name(c) for c in train_dataset.classes]
    num_classes = len(class_names)
    print(f"✅ Классы ({num_classes}): {class_names}")

    with open("class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- МОДЕЛЬ (Large!) ---
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    # 1) замораживаем всё
    for param in model.parameters():
        param.requires_grad = False

    # 2) размораживаем последние 5 блоков features (у Large их 17)
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    # 3) новый классификатор (вход 960, а не 576!)
    num_features = model.classifier[0].in_features   # 960 для Large
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🖥️  Используем: {device}")
    print(f"📐 Вход классификатора: {num_features} → {num_classes}")

    # --- ОПТИМИЗАТОР: два lr ---
    optimizer = optim.Adam([
        {"params": model.features[-5:].parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(),    "lr": LR_HEAD},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # --- VAL ---
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
                "arch": "mobilenet_v3_large",   # ← маркер архитектуры
            }, "fabric_model_best.pth")
            print(f"💾 Сохранена лучшая модель (Val Acc: {val_acc:.2f}%)")

    # --- TEST ---
    checkpoint = torch.load("fabric_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"✅ Test Accuracy: {100 * test_correct / test_total:.2f}%")
    print(f"📊 Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()