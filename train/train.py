import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import time
import random
seed = 42
random.seed(seed)
torch.manual_seed(seed)
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------
# 1️⃣ Crop vertical 15%
# --------------------------------------------------

def crop_vertical_15_percent(img):
    width, height = img.size
    top = int(0.15 * height)
    bottom = int(0.85 * height)
    return img.crop((0, top, width, bottom))

# --------------------------------------------------
# 2️⃣ Transformations (propre, sans double resize)
# --------------------------------------------------

mobilenet_weights = models.MobileNet_V2_Weights.DEFAULT

# ImageNet normalization (standard, stable)
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

mobilenet_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_transform = transforms.Compose([
    transforms.Lambda(crop_vertical_15_percent),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    mobilenet_preprocess,
])

val_transform = transforms.Compose([
    transforms.Lambda(crop_vertical_15_percent),
    transforms.Resize((224, 224)),
    mobilenet_preprocess,
])

# --------------------------------------------------
# 3️⃣ Charger dataset (split depuis fichiers)
# --------------------------------------------------

from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from torch.utils.data import Dataset

    
dataset = datasets.ImageFolder("dataset", transform=None)

from sklearn.model_selection import train_test_split

# Tous les indices
all_indices = list(range(len(dataset)))

# Tous les labels
all_labels = [label for _, label in dataset.samples]

# Split stratifié (garde la proportion des classes)
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.18,   # ~18% validation
    stratify=all_labels,
    random_state=42
)

train_dataset = datasets.ImageFolder("dataset", transform=train_transform)
val_dataset   = datasets.ImageFolder("dataset", transform=val_transform)

train_dataset = Subset(train_dataset, train_idx)
val_dataset   = Subset(val_dataset, val_idx)

# Appliquer transforms
from torch.utils.data import Dataset

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Classes détectées :", dataset.classes)
print("Train images:", len(train_dataset), "| Val images:", len(val_dataset))

# --------------------------------------------------
# 4️⃣ Modèle MobileNetV2 (API propre)
# --------------------------------------------------

model = models.mobilenet_v2(weights=mobilenet_weights)

for param in model.features.parameters():
    param.requires_grad = False

# Dé-geler les 2 derniers blocs (fine-tuning léger)
for param in model.features[-2:].parameters():
    param.requires_grad = True


num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(dataset.classes))

# --------------------------------------------------
# 5️⃣ Loss & Optimizer
# --------------------------------------------------

from collections import Counter
import torch

train_targets = [dataset.samples[i][1] for i in train_idx]
counts = Counter(train_targets)

num_classes = len(dataset.classes)
weights = torch.tensor([1.0 / counts[i] for i in range(num_classes)], dtype=torch.float)
weights = weights / weights.sum() * num_classes

criterion = nn.CrossEntropyLoss(weight=weights)
print("Class weights:", {dataset.classes[i]: float(weights[i]) for i in range(num_classes)})

optimizer = optim.Adam([
    {"params": model.classifier.parameters(), "lr": 1e-3},
    {"params": model.features[-2:].parameters(), "lr": 1e-4},
])


# --------------------------------------------------
# 6️⃣ Entraînement
# --------------------------------------------------

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total if total > 0 else float("nan")

epochs = 10
best_val = 0.0

for epoch in range(epochs):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    epoch_time = time.time() - t0

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Time: {epoch_time:.1f}s",
        flush=True
    )

    val_acc = evaluate(model, val_loader)
    print(f"Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_val:
        best_val = val_acc
        best_checkpoint = {
          "model_state": model.state_dict(),
          "classes": dataset.classes,
          "img_size": 224,
          "crop_vertical": 0.15,
          "mean": mean,
          "std": std,
    }
    torch.save(best_checkpoint, "bjj_model_best.pth")
    print("✅ Saved best model (checkpoint)")

# --------------------------------------------------
# 7️⃣ Validation
# --------------------------------------------------

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if total == 0:
    print("Validation: 0 images (val split vide). Skipping validation.")
    val_accuracy = float("nan")
else:
    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# --------------------------------------------------
# 7bis️⃣ Matrice de confusion + rapport par classe
# --------------------------------------------------

all_true = []
all_pred = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

labels = list(range(len(dataset.classes)))
cm = confusion_matrix(all_true, all_pred, labels=labels)
print("\nMatrice de confusion (Validation) :")
print("Lignes = vrai label, Colonnes = prédiction")
print("Classes :", dataset.classes)
print(cm)

print("\nRapport par classe (Validation) :")
labels = list(range(len(dataset.classes)))
print(classification_report(
    all_true, all_pred,
    labels=labels,
    target_names=dataset.classes,
    digits=3,
    zero_division=0
))


# --------------------------------------------------
# 8️⃣ Sauvegarde
# --------------------------------------------------

checkpoint = {
    "model_state": model.state_dict(),
    "classes": dataset.classes,
    "img_size": 224,
    "crop_vertical": 0.15,
    "mean": mean, 
    "std": std,
}
print(f"Best val = {best_val:.2f}% — modèle best déjà sauvegardé dans bjj_model_best.pth")

best = torch.load("bjj_model_best.pth", map_location="cpu")
model.load_state_dict(best["model_state"])
model.eval()
print("Loaded best model for test.jpg")

# --------------------------------------------------
# 9️⃣ Test sur image externe
# --------------------------------------------------

model.eval()

img_path = "test.jpg"
image = Image.open(img_path)

image = val_transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)

# Top-2
top2_prob, top2_idx = torch.topk(probs, k=2, dim=1)
top2 = [(dataset.classes[int(i)], float(p)) for p, i in zip(top2_prob[0], top2_idx[0])]

print("Top-2 :", top2)
print("Probabilités :", probs)

# Seuil de confiance
threshold = 0.45
best_class, best_prob = top2[0]

if best_prob < threshold:
    print(f"Résultat: INCERTAIN (best={best_class} {best_prob:.3f})")
else:
    print(f"Résultat: {best_class} ({best_prob:.3f})")

