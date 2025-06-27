import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# ========== Device Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ========== Haar Cascades ==========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ========== Face Alignment ==========
def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    if len(eyes) >= 2:
        eye1, eye2 = eyes[:2]
        eye_center1 = (eye1[0] + eye1[2]//2, eye1[1] + eye1[3]//2)
        eye_center2 = (eye2[0] + eye2[2]//2, eye2[1] + eye2[3]//2)

        if eye_center2[0] < eye_center1[0]:
            eye_center1, eye_center2 = eye_center2, eye_center1

        dx = eye_center2[0] - eye_center1[0]
        dy = eye_center2[1] - eye_center1[1]
        angle = np.degrees(np.arctan2(dy, dx))

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        aligned = cv2.warpAffine(face, M, (w, h))
        return aligned
    return face

# ========== Dataset ==========
class AlignedFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        for label, folder in enumerate(['real', 'fake']):
            path = os.path.join(root_dir, folder)
            for img in os.listdir(path):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.paths.append(os.path.join(path, img))
                    self.labels.append(label)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        aligned = align_face(img)
        if aligned is None:
            aligned = img  # fallback
        img_pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label

    def __len__(self):
        return len(self.paths)

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ========== Load Data ==========
dataset_path = "D:\\Dev\\Code10Thrive\\image_dataset"  # 🔁 Update this
dataset = AlignedFaceDataset(dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== Model ==========
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# ========== Training Setup ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
epochs = 20

losses, accuracies, times = [], [], []

# ========== Training Loop ==========
for epoch in range(epochs):
    model.train()
    start = time.time()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    duration = time.time() - start

    losses.append(epoch_loss)
    accuracies.append(acc)
    times.append(duration)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.2f}% | Time: {duration:.2f}s")

# ========== Save Trained Model ==========
torch.save(model.state_dict(), "D:\\Dev\\Code10Thrive\\DeepFake_Detector\\Models\\Image\\deepfake_image_model1_stateDic.pth")
torch.save(model, "D:\\Dev\\Code10Thrive\\DeepFake_Detector\\Models\\Image\\deepfake_image_model1.pth")
print("✅ Model saved.")

# ========== Plotting ==========
plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)
plt.plot(losses, marker='o', label='Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(accuracies, marker='x', color='green', label='Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(times, marker='s', color='orange', label='Time')
plt.title("Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Seconds")
plt.grid(True)

plt.tight_layout()
plt.show()
