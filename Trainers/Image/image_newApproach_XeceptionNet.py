import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# === Face Alignment ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if len(faces) == 0: return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

# === Dataset ===
class AlignedFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths, self.labels = [], []
        self.transform = transform
        for label, folder in enumerate(['real', 'fake']):
            path = os.path.join(root_dir, folder)
            for img in os.listdir(path):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.paths.append(os.path.join(path, img))
                    self.labels.append(label)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        aligned = align_face(img)
        if aligned is None:
            aligned = img  # fallback to original image if no face detected

        img_pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, self.labels[idx]

    def __len__(self): return len(self.paths)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((299, 299)),   # Xception expects 299x299
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load Dataset ===
dataset = AlignedFaceDataset("D:/Projects/PythonProject/Dataset/Images", transform=transform)

# Split into Train (70%), Val (20%), Test (10%)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=24, shuffle=True)
val_loader = DataLoader(val_set, batch_size=24, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# === Model ===
model = timm.create_model("xception", pretrained=True, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
epochs = 8

train_losses, val_losses, train_accuracies, val_accuracies, times = [], [], [], [], []

# === Training Loop ===
for epoch in range(epochs):
    start = time.time()
    model.train()
    epoch_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(train_acc)

    # === Validation ===
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    duration = time.time() - start
    times.append(duration)

    print(f"Epoch {epoch+1}: TrainLoss={epoch_loss:.4f}, TrainAcc={train_acc:.2f}%, "
          f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.2f}%, Time={duration:.2f}s")

torch.save(model.state_dict(), "deepfake_xception_stateDict.pth")
torch.save(model,"deepfake_xception.pth")
print("✅ Xception model saved.")

# === Graphs ===
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend(); plt.title("Loss over Epochs"); plt.grid(True)

plt.subplot(1,3,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.legend(); plt.title("Accuracy over Epochs"); plt.grid(True)

plt.subplot(1,3,3)
plt.plot(times, marker='o')
plt.title("Time per Epoch (s)")
plt.grid(True)

plt.tight_layout()
plt.show()
