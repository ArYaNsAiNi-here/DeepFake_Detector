# Deepfake Audio Detection with Enhanced CNN, GPU Monitoring, Training Stats, and Timing Logs

import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import GPUtil
import time
from torch.utils.data import Dataset, DataLoader

class AudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, sr=16000, duration=3):
        self.paths = []
        self.labels = []
        self.sr = sr
        self.duration = duration
        self.max_len = sr * duration
        valid_exts = ('.wav', '.mp3', '.flac')

        for label, folder in enumerate(['real', 'fake']):
            full_path = os.path.join(root_dir, folder)
            for f in os.listdir(full_path):
                if any(ext in f.lower() for ext in valid_exts):
                    self.paths.append(os.path.join(full_path, f))
                    self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        waveform = waveform.mean(dim=0)
        if waveform.shape[0] < self.max_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.shape[0]))
        else:
            waveform = waveform[:self.max_len]

        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return log_mel.unsqueeze(0), label

class DeepFakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)

# Setup
root_dir = 'D:\\Dev\\Code10Thrive\\audio_dataset'
dataset = AudioSpectrogramDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
if device.type == 'cuda':
    print("GPU Name:", torch.cuda.get_device_name(0))

model = DeepFakeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

assert next(model.parameters()).is_cuda, "Model is NOT on GPU!"

# Training with stats, GPU monitoring, timing, and debug printing
num_epochs = 20
train_losses = []
train_accuracies = []
epoch_times = []

def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name} | Load: {gpu.load*100:.1f}% | Mem: {gpu.memoryUsed}/{gpu.memoryTotal} MB")

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        print(f"Batch device: {inputs.device}, Labels device: {labels.device}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    epoch_duration = end_time - start_time
    epoch_times.append(epoch_duration)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_duration:.2f}s")
    if device.type == 'cuda':
        print_gpu_usage()

# Save model
torch.save(model, "deepfake_audio_full_model.pth")

# Combined Plot
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Loss', marker='o')
plt.plot(train_accuracies, label='Accuracy (%)', marker='x')
plt.plot(epoch_times, label='Epoch Time (s)', marker='s')
plt.xlabel('Epoch')
plt.title('Training Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
