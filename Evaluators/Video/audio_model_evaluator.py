import os
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn

# === Audio Preprocess ===
def preprocess_audio(file_path, sample_rate=16000, duration=3):
    max_len = sample_rate * duration
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0)

    if waveform.shape[0] < max_len:
        waveform = F.pad(waveform, (0, max_len - waveform.shape[0]))
    else:
        waveform = waveform[:max_len]

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return log_mel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

# === Define Model ===
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

# === Load Model ===
def load_model(model_path, device):
    model = DeepFakeCNN()
    state_dict = torch.load(model_path, map_location=device)

    if "net.0.weight" in state_dict:
        model.load_state_dict(state_dict)
    else:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict["net." + k] = v
        model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    return model

# === Prediction ===
def predict_deepfake(audio_file, model, device):
    input_tensor = preprocess_audio(audio_file).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return "Real" if pred == 0 else "Fake", confidence

# === Main ===
if __name__ == "__main__":
    torchaudio.set_audio_backend("soundfile")
    original_file = "C:\\Users\\Asus\\Downloads\\Test_Data\\audio\\wav_fake\\fake1.wav"
    model_path = "D:\\Projects\\PythonProject\\DeepFake_Detector\\Models\\Audio\\deepfake_audio_model1_stateDict.pth"

    if not os.path.exists(original_file):
        print("File does not exist.")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    label, conf = predict_deepfake(original_file, model, device)
    print(f"\nPrediction: {label} (Accuracy: {conf*100:.2f}%)")
