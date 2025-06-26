import os
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from pydub import AudioSegment

# === Step 1: Convert to WAV ===
def convert_to_wav(input_path, output_path, target_sr=16000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sr)
    audio.export(output_path, format="wav")
    return output_path

# === Step 2: Audio Preprocessing ===
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

# === Step 3: Define Model ===
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

# === Step 4: Load Trained Model ===
def load_model(model_path, device):
    model = DeepFakeCNN()
    state_dict = torch.load(model_path, map_location=device)

    # If trained with torch.save(model.state_dict())
    if "net.0.weight" in state_dict:
        model.load_state_dict(state_dict)
    else:  # If keys have no 'net.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict["net." + k] = v
        model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    return model

# === Step 5: Prediction Function ===
def predict_deepfake(audio_file, model, device):
    input_tensor = preprocess_audio(audio_file).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return "Real" if pred == 0 else "Fake", confidence

# === Step 6: Main Script ===
if __name__ == "__main__":
    original_file = "D:\\Dev\\Code10Thrive\\2\\for-original\\for-original\\training\\real\\file12.wav"  # 🔁 CHANGE THIS
    temp_wav = "temp_audio.wav"
    model_path = "D:\\Dev\\Code10Thrive\\DeepFake_Detector\\Models\\deepfake_audio_model1_stateDict.pth"

    if not os.path.exists(original_file):
        print("File does not exist.")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convert_to_wav(original_file, temp_wav)
    model = load_model(model_path, device)

    label, conf = predict_deepfake(temp_wav, model, device)
    print(f"\nPrediction: {label} (Accuracy : {conf*100:.2f}%)")

    # Clean up temp file
    os.remove(temp_wav)