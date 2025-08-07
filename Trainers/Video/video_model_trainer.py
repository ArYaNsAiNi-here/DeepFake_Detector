import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = 'D:\\Projects\\PythonProject\\Dataset\\Video'
MODEL_STATE_DICT_PATH = "D:\Projects\PythonProject\DeepFake_Detector\Models\Video\deepfake_video_model_stateDict.pth"
FULL_MODEL_PATH = "D:\Projects\PythonProject\DeepFake_Detector\Models\Video\deepfake_video_model.pth"
NUM_CLASSES = 1
BATCH_SIZE = 5
EPOCHS = 5
LEARNING_RATE = 0.0001
IMG_SIZE = 224
MAX_FRAMES_PER_VIDEO = 24


# --- 1. Custom Dataset for Videos ---
class VideoDataset(Dataset):
    def __init__(self, video_files, labels, transform=None, max_frames=MAX_FRAMES_PER_VIDEO):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        return frames, torch.tensor(label, dtype=torch.float32)

    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * self.max_frames

        frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        cap.release()
        return frames


# --- 2. Model Definition ---
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(VideoClassifier, self).__init__()
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        num_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(num_features, num_classes))

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)
        frame_features = self.cnn(x).view(batch_size, num_frames, -1)
        return torch.mean(frame_features, dim=1).squeeze(1)


# --- 3. Main Training & Evaluation Logic ---
def main():
    print("Preparing data...")
    all_video_files, all_labels = [], []
    for label, category in enumerate(['Real', 'Fake']):
        category_dir = os.path.join(DATA_DIR, category)
        if not os.path.exists(category_dir): raise FileNotFoundError(f"Directory not found: {category_dir}")
        videos = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if
                  f.endswith(('.mp4', '.avi', '.mov'))]
        all_video_files.extend(videos)
        all_labels.extend([label] * len(videos))

    train_files, val_files, train_labels, val_labels = train_test_split(all_video_files, all_labels, test_size=0.1,
                                                                        random_state=42, stratify=all_labels)

    data_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset, val_dataset = VideoDataset(train_files, train_labels, transform=data_transform), VideoDataset(
        val_files, val_labels, transform=data_transform)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=4), DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                                     num_workers=4)
    print(f"Training on {len(train_dataset)} videos, validating on {len(val_dataset)} videos.")

    model = VideoClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'time': []}

    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # <<< PROGRESS BAR FOR THE TRAINING LOOP >>>
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]"):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct_predictions = 0.0, 0
        with torch.no_grad():
            # <<< PROGRESS BAR FOR THE VALIDATION LOOP >>>
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Validation]"):
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * videos.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct_predictions += (preds == labels.byte()).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = (correct_predictions / len(val_loader.dataset)) * 100
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% | Time: {epoch_time:.2f}s")

        history['loss'].append(epoch_loss);
        history['val_loss'].append(val_epoch_loss);
        history['accuracy'].append(val_accuracy);
        history['time'].append(epoch_time)

    print("\nTraining finished.")

    print("Saving model...")
    torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
    torch.save(model, FULL_MODEL_PATH)
    print("Model saved.")

    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['loss'], 'o-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['accuracy'], 'x-', color='green', label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['time'], 's-', color='orange', label='Time per Epoch')
    plt.title("Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()