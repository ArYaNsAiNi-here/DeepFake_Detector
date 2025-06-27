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
from pydub import AudioSegment

# === Utility Functions ===
def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

def is_valid_audio(file):
    valid_exts = ('.wav', '.mp3')
    return any(ext in file.lower() for ext in valid_exts)

# === Dataset without Class ===
def load_dataset(root_dir, sr=16000, duration=3):
    paths = []
    labels = []
    max_len = sr * duration

    for label, folder in enumerate(['real', 'fake']):
        full_path = os.path.join(root_dir, folder)
        for f in os.listdir(full_path):
            if is_valid_audio(f):
                filepath = os.path.join(full_path, f)
                if not f.endswith(".wav"):
                    wav_path = filepath + ".converted.wav"
                    convert_to_wav(filepath, wav_path)
                    filepath = wav_path
                paths.append(filepath)
                labels.append(label)
    return paths, labels, max_len

def get_item(path, label, max_len):
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

    waveform = waveform.mean(dim=0)
    if waveform.shape[0] < max_len:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))
    else:
        waveform = waveform[:max_len]

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return log_mel.unsqueeze(0), label

# === Model Definition ===
def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

# === GPU Monitor ===
def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name} | Load: {gpu.load*100:.1f}% | Mem: {gpu.memoryUsed}/{gpu.memoryTotal} MB")

# === Training ===
def train_model():
    root_dir = 'D:/Dev/Code10Thrive/audio_dataset'
    paths, labels, max_len = load_dataset(root_dir)
    data = list(zip(paths, labels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    assert next(model.parameters()).is_cuda, "Model is NOT on GPU!"

    num_epochs = 20
    batch_size = 32
    train_losses, train_accuracies, epoch_times = [], [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            inputs, targets = [], []
            for path, label in batch_data:
                x, y = get_item(path, label, max_len)
                if x is not None:
                    inputs.append(x)
                    targets.append(y)

            if not inputs:
                continue

            inputs = torch.stack(inputs).to(device)
            targets = torch.tensor(targets).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        end_time = time.time()
        epoch_times.append(end_time - start_time)
        epoch_loss = running_loss / (len(data) // batch_size + 1)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {end_time - start_time:.2f}s")
        if device.type == 'cuda':
            print_gpu_usage()

    torch.save(model.state_dict(), "./Models/deepfake_audio_model1_stateDict.pth")
    torch.save(model, "./Models/deepfake_audio_model1.pth")

    # === Plot 1: Loss ===
    plt.figure()
    plt.plot(train_losses, label='Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot 2: Accuracy ===
    plt.figure()
    plt.plot(train_accuracies, label='Accuracy', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot 3: Epoch Time ===
    plt.figure()
    plt.plot(epoch_times, label='Epoch Time (s)', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Epoch Duration')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()