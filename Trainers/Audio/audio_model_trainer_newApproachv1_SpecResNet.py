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
from tqdm import tqdm
from torch import amp
import warnings

# Suppress the UserWarning from torchaudio.load as it's a deprecation notice for a future release
warnings.filterwarnings('ignore', category=UserWarning)

# === Configuration ===
# IMPORTANT: Update ROOT_DIR to the correct path of your dataset on your system
ROOT_DIR = 'D:\\Projects\\PythonProject\\Dataset\\release_in_the_wild' # Make sure this path is correct
MODEL_STATE_DICT_PATH = "D:/Projects/PythonProject/DeepFake_Detector/Models/Audio/deepfake_audio_specresnet.pth"
FULL_MODEL_PATH = "D:/Projects/PythonProject/DeepFake_Detector/Models/Audio/deepfake_audio_specresnet_stateDict.pth"

# Audio processing parameters
SAMPLE_RATE = 16000  # Standard sample rate for many audio tasks
DURATION = 5         # Duration of audio clips in seconds
MAX_LEN = SAMPLE_RATE * DURATION # Max number of samples per clip

# Training hyperparameters
BATCH_SIZE = 48
NUM_EPOCHS = 20
LEARNING_RATE = 0.0015

# Determine device for training (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Utility Functions ===
def convert_to_wav(input_path, output_path):
    """Converts any audio file supported by pydub to WAV format."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception:
        # Silently fail for now, or add print(f"Could not convert {input_path}: {e}") for debugging
        return False

# === Dataset Preparation for Structured Directories ===
def get_filepaths_and_labels(split_dir):
    """
    Collects file paths and assigns labels (0 for 'real', 1 for 'fake')
    from structured 'real' and 'fake' subdirectories within a split.
    Handles non-WAV files by converting them.
    """
    paths, labels = [], []
    for label, cls in enumerate(['real', 'fake']): # 0 for real, 1 for fake
        class_dir = os.path.join(split_dir, cls)
        if not os.path.exists(class_dir):
            continue # Skip if class directory doesn't exist for this split

        for f in os.listdir(class_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                filepath = os.path.join(class_dir, f)
                # Convert to WAV if not already, to ensure torchaudio compatibility
                if not f.lower().endswith(".wav"):
                    wav_path = os.path.splitext(filepath)[0] + ".wav"
                    if convert_to_wav(filepath, wav_path):
                        paths.append(wav_path)
                        labels.append(label)
                else:
                    paths.append(filepath)
                    labels.append(label)
    return paths, labels

# === Mel Spectrogram Transform (initialized on CPU to work with DataLoader) ===
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
) # No .to(device) here, transform will be applied to CPU tensors

# === PyTorch Dataset Class ===
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
            # Resample if sample rate doesn't match config
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            # Convert stereo to mono by taking the mean across channels
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or truncate waveform to a fixed length (MAX_LEN)
            if waveform.shape[1] < MAX_LEN:
                waveform = torch.nn.functional.pad(waveform, (0, MAX_LEN - waveform.shape[1]))
            else:
                waveform = waveform[:, :MAX_LEN]

            # Apply Mel spectrogram transformation (waveform is on CPU here)
            mel_spec = self.transform(waveform)
            return mel_spec, torch.tensor(label, dtype=torch.long)
        except Exception:
            # Return dummy tensors and label for corrupted/unreadable files.
            # This prevents DataLoader from crashing but might need more robust handling for large datasets.
            return torch.zeros((1, 128, 501)), torch.tensor(0, dtype=torch.long)

# === Model Architecture: Spec-ResNet ===
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
        return self.relu(out)

class SpecResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(SpecResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Corrected indexing for num_blocks
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
        return self.linear(out)

def create_spec_resnet():
    # Defines a simple ResNet-like architecture with 4 stages
    return SpecResNet(ResidualBlock, [2, 2, 2, 2])

# === GPU Monitor ===
def print_gpu_usage():
    """Prints current GPU utilization and memory usage."""
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name} | Load: {gpu.load*100:.1f}% | Mem: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
    except Exception:
        pass # Silently pass if GPUtil is not installed or no GPU is found

# === Main Training and Validation Function ===
def train_model():
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))

    print("Loading and preparing dataset...")
    # Define paths for each data split
    train_dir = os.path.join(ROOT_DIR, "train")
    val_dir = os.path.join(ROOT_DIR, "val")
    test_dir = os.path.join(ROOT_DIR, "test") # If you plan to use a test set for final evaluation

    # Get file paths and labels for each split
    train_paths, train_labels = get_filepaths_and_labels(train_dir)
    val_paths, val_labels = get_filepaths_and_labels(val_dir)
    # test_paths, test_labels = get_filepaths_and_labels(test_dir) # Uncomment if you need the test set

    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_paths, train_labels, mel_transform)
    val_dataset = AudioDataset(val_paths, val_labels, mel_transform)
    # test_dataset = AudioDataset(test_paths, test_labels, mel_transform) # Uncomment if you need the test set

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True) # Uncomment if you need the test set

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # Initialize model, loss function, and optimizer
    model = create_spec_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize GradScaler for Automatic Mixed Precision (AMP)
    # Correct usage: provide device_type and enabled status
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'time': []}

    # === Training Loop ===
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train() # Set model to training mode
        running_loss, correct, total = 0.0, 0, 0

        # Iterate over batches in the training loader
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"):
            # Move input tensors and labels to the training device (GPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Use autocast for mixed precision training
            # Correct usage: provide device_type, optionally dtype, and enabled status
            with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad() # Clear gradients from previous step
            scaler.scale(loss).backward() # Scale loss and perform backward pass
            scaler.step(optimizer) # Update model parameters
            scaler.update() # Update the scaler for the next iteration

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # === Validation Loop ===
        model.eval() # Set model to evaluation mode
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(): # Disable gradient calculations for validation
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
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

        # Print epoch summary
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")
        if device.type == 'cuda':
            print_gpu_usage() # Display GPU usage if on CUDA

    # === Save Model and Plot Training History ===
    torch.save(model.state_dict(), MODEL_STATE_DICT_PATH) # Save model's learned parameters
    torch.save(model, FULL_MODEL_PATH) # Save the entire model (architecture + parameters)
    print(f"Model state saved to {MODEL_STATE_DICT_PATH}")

    # Plotting training history
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1) # Loss plot
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'o-', label='Val Loss')
    plt.title("Loss"); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 2) # Accuracy plot
    plt.plot(epochs_range, history['train_acc'], 'x-', label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], 'x-', label='Val Acc')
    plt.title("Accuracy"); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3) # Time per epoch plot
    plt.plot(epochs_range, history['time'], 's-', label='Epoch Time')
    plt.title("Time per Epoch"); plt.legend(); plt.grid(True)

    plt.tight_layout() # Adjust plot to prevent overlapping titles/labels
    plt.show() # Display the plots

if __name__ == "__main__":
    train_model()
