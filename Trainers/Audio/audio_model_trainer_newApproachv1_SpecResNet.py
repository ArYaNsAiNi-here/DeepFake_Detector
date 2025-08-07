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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# === Configuration ===
ROOT_DIR = 'D:/Dev/Code10Thrive/audio_dataset'
FULL_MODEL_PATH = "D:/Projects/PythonProject/DeepFake_Detector/Models/Audio/deepfake_audio_specresnet_stateDict.pth"
MODEL_STATE_DICT_PATH = "D:/Projects/PythonProject/DeepFake_Detector/Models/Audio/deepfake_audio_specresnet.pth"
SAMPLE_RATE = 16000
DURATION = 5  # seconds
MAX_LEN = SAMPLE_RATE * DURATION
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# === Utility Functions ===
def convert_to_wav(input_path, output_path):
    """Converts any audio file supported by pydub to WAV format."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Could not convert {input_path}: {e}")
        return False


# === 1. PyTorch Dataset for Efficient Data Loading ===
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(filepath)
            # Resample if necessary and convert to mono
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or truncate to a fixed length
            if waveform.shape[1] < MAX_LEN:
                waveform = torch.nn.functional.pad(waveform, (0, MAX_LEN - waveform.shape[1]))
            else:
                waveform = waveform[:, :MAX_LEN]

            # Apply Mel spectrogram transformation
            mel_spec = self.transform(waveform)
            return mel_spec, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading or processing {filepath}: {e}")
            # Return a dummy tensor and label to avoid crashing the DataLoader
            return torch.zeros((1, 128, 501)), torch.tensor(0, dtype=torch.long)


# === 2. Advanced Model Architecture: Spec-ResNet ===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpecResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(SpecResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_spec_resnet():
    return SpecResNet(ResidualBlock, [2, 2, 2, 2])


# === GPU Monitor ===
def print_gpu_usage():
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(
                f"GPU {gpu.id}: {gpu.name} | Load: {gpu.load * 100:.1f}% | Mem: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
    except Exception as e:
        print(f"Could not get GPU details: {e}")


# === Main Training and Validation Function ===
def train_model():
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))

    # --- Data Preparation ---
    print("Loading and preparing dataset...")
    paths, labels = [], []
    for label, folder in enumerate(['real', 'fake']):
        full_path = os.path.join(ROOT_DIR, folder)
        for f in os.listdir(full_path):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                filepath = os.path.join(full_path, f)
                if not f.lower().endswith(".wav"):
                    wav_path = os.path.splitext(filepath)[0] + ".wav"
                    if convert_to_wav(filepath, wav_path):
                        paths.append(wav_path)
                        labels.append(label)
                else:
                    paths.append(filepath)
                    labels.append(label)

    # Split data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels)

    # Define Mel spectrogram transformation
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
    ).to(device)

    train_dataset = AudioDataset(train_paths, train_labels, mel_transform)
    val_dataset = AudioDataset(val_paths, val_labels, mel_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- Model, Optimizer, Loss ---
    model = create_spec_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # For mixed precision

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'time': []}

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # --- Training Loop ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- Validation Loop ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - start_time
        history['time'].append(epoch_time)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")
        if device.type == 'cuda':
            print_gpu_usage()

    # --- Save and Plot ---
    torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
    torch.save(model, FULL_MODEL_PATH)
    print(f"Model state saved to {MODEL_STATE_DICT_PATH}")

    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1);
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss');
    plt.plot(epochs_range, history['val_loss'], 'o-', label='Val Loss');
    plt.title("Loss");
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 3, 2);
    plt.plot(epochs_range, history['train_acc'], 'x-', label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], 'x-', label='Val Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['time'], 's-', label='Epoch Time')
    plt.title("Time per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model()