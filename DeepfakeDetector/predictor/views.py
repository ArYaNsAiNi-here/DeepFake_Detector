import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio
from PIL import Image
import cv2
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone

# === Import Database Model ===
from .models import UploadHistory

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Paths ---
IMAGE_MODEL_PATH = "D:\\Projects\\PythonProject\\DeepFake_Detector\\Models\\Image\\deepfake_image_model1_stateDict.pth"
VIDEO_MODEL_PATH = "D:\\Projects\\PythonProject\\DeepFake_Detector\\Models\\Video\\deepfake_video_model_stateDict.pth"
AUDIO_MODEL_PATH = "D:\\Projects\\PythonProject\\DeepFake_Detector\\Models\\Audio\\deepfake_audio_model1_stateDict.pth"

IMG_SIZE = 224
MAX_FRAMES_PER_VIDEO = 16
SAMPLE_RATE = 16000
AUDIO_DURATION = 3
MAX_AUDIO_LEN = SAMPLE_RATE * AUDIO_DURATION

# === Model Architectures ===
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(VideoClassifier, self).__init__()
        self.cnn = models.efficientnet_b0(weights=None)
        num_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)
        frame_features = self.cnn(x).view(batch_size, num_frames, -1)
        video_prediction = torch.mean(frame_features, dim=1)
        return video_prediction.squeeze(1)

def create_audio_model():
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

# === Load Models ===
image_model = models.resnet50(weights=None)
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2)
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
image_model.to(device).eval()

video_model = VideoClassifier(num_classes=1)
video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=device))
video_model.to(device).eval()

audio_model = create_audio_model()
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
audio_model.to(device).eval()

# === Handlers ===
def handle_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = image_model(img)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
    return predicted.item(), confidence.item()

def handle_audio(path):
    try:
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0)
        if waveform.shape[0] < MAX_AUDIO_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, MAX_AUDIO_LEN - waveform.shape[0]))
        else:
            waveform = waveform[:MAX_AUDIO_LEN]
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)(waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = audio_model(log_mel_spec)
            probabilities = F.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"Error processing audio: {e}")
        return -1, 0

def handle_video(path):
    try:
        cap = cv2.VideoCapture(path)
        frames, total_frames = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frames_tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = video_model(frames_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            confidence = probability if prediction == 1 else 1 - probability
        return prediction, confidence
    except Exception as e:
        print(f"Error processing video: {e}")
        return -1, 0

# === Django Views ===
def upload_page(request):
    return render(request, 'predictor/upload.html')

def detect_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    file = request.FILES.get('media')
    if not file:
        return JsonResponse({'error': 'No file provided'}, status=400)

    # Save file permanently in MEDIA_ROOT/uploads/
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # Predict
    ext = os.path.splitext(file.name)[1].lower()
    prediction, confidence, media_type = -1, 0, "unknown"

    if ext in ['.jpg', '.jpeg', '.png']:
        prediction, confidence = handle_image(file_path)
        media_type = "image"
    elif ext in ['.wav', '.mp3', '.flac']:
        prediction, confidence = handle_audio(file_path)
        media_type = "audio"
    elif ext in ['.mp4', '.avi', '.mov']:
        prediction, confidence = handle_video(file_path)
        media_type = "video"
    else:
        return JsonResponse({'text': 'Unsupported file type.', 'class': 'error'}, status=415)

    # Save DB record
    UploadHistory.objects.create(
        user=request.user if request.user.is_authenticated else None,
        file=os.path.relpath(file_path, settings.MEDIA_ROOT),
        media_type=media_type,
        prediction="Deepfake" if prediction == 1 else "Real" if prediction == 0 else "Error",
        confidence=confidence,
        uploaded_at=timezone.now()
    )

    # Response
    if prediction == 1:
        return JsonResponse({'text': '⚠️ Likely DEEPFAKE', 'class': 'fake', 'confidence': f"{confidence*100:.2f}%"})
    elif prediction == 0:
        return JsonResponse({'text': '✅ Appears REAL', 'class': 'real', 'confidence': f"{confidence*100:.2f}%"})
    else:
        return JsonResponse({'text': '❌ Could not process file.', 'class': 'error'})
