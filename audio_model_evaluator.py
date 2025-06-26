import torch
import torchaudio
import torch.nn.functional as F
import os
from torch import nn

# === Define Model Architecture ===
class DeepFakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, x):
        return self.net(x)

# === Load Model with Weights Only ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepFakeCNN()
model.load_state_dict(torch.load("D:\\Dev\\Code10Thrive\\DeepFake_Detector\\Models\\deepfake_audio_model_enhanced_1.pth", map_location=device))
model.eval()
model.to(device)

# === Preprocessing Function ===
def preprocess_audio(file_path, sample_rate=16000, duration=3):
    max_len = sample_rate * duration

    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0)  # Convert to mono

    if waveform.shape[0] < max_len:
        waveform = F.pad(waveform, (0, max_len - waveform.shape[0]))
    else:
        waveform = waveform[:max_len]

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return log_mel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

# === Prediction Function ===
def predict(file_path):
    input_tensor = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()
    label = "Real" if predicted == 0 else "Fake"
    return label, confidence

# === Example Usage ===
if __name__ == "__main__":
    test_file = "D:\\Dev\\Code10Thrive\\2\\for-original\\for-original\\testing\\fake\\file1.wav"  # Change this
    if not os.path.exists(test_file):
        print("Audio file not found! Please check the path.")
    else:
        label, conf = predict(test_file)
        print(f"Prediction: {label} (Confidence: {conf:.2f})")
