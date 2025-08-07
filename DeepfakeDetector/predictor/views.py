import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import the functional module
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio
from PIL import Image
import cv2
import numpy as np
from django.shortcuts import render
from django.conf import settings
from .forms import MediaUploadForm

# === Configuration (remains the same) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MODEL_PATH = "D:\Projects\PythonProject\DeepFake_Detector\Models\Image\deepfake_image_model1_stateDict.pth"
VIDEO_MODEL_PATH = "D:\Projects\PythonProject\DeepFake_Detector\Models\Video\deepfake_video_model_stateDict.pth"
AUDIO_MODEL_PATH = "D:\Projects\PythonProject\DeepFake_Detector\Models\Audio\deepfake_audio_model1_stateDict.pth"

# ... other configs ...
IMG_SIZE = 224
MAX_FRAMES_PER_VIDEO = 32
SAMPLE_RATE = 16000
DURATION = 5
MAX_LEN = SAMPLE_RATE * DURATION


# === Model Architectures (remain the same) ===
class VideoClassifier(nn.Module):
    # ... (no changes needed)
    def __init__(self, num_classes=2):
        super(VideoClassifier, self).__init__()
        self.cnn = models.efficientnet_b0(weights=None)
        num_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(num_features, num_classes))

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)
        frame_features = self.cnn(x)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        video_prediction = torch.mean(frame_features, dim=1)
        return video_prediction


class ResidualBlock(nn.Module):
    # ... (no changes needed)
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SpecResNet(nn.Module):
    # ... (no changes needed)
    def __init__(self, block, num_blocks, num_classes=2):
        super(SpecResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
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
        out = self.layer1(out);
        out = self.layer2(out);
        out = self.layer3(out);
        out = self.layer4(out)
        out = self.avg_pool(out).view(out.size(0), -1)
        return self.linear(out)


# === Model Loading (remains the same) ===
# ... (all model loading code is the same)
# --- Image Model ---
image_model = models.resnet50(weights=None)
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2)
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
image_model.to(device)
image_model.eval()
print("Image model loaded successfully.")
# --- Video Model ---
video_model = VideoClassifier(num_classes=2)
video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=device))
video_model.to(device)
video_model.eval()
print("Video model loaded successfully.")
# --- Audio Model ---
audio_model = SpecResNet(ResidualBlock, [2, 2, 2, 2], num_classes=2)
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
audio_model.to(device)
audio_model.eval()
print("Audio model loaded successfully.")


# === MODIFIED: Handlers now return (prediction, confidence) ===

def handle_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = image_model(img)
        # Get probabilities using softmax
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
    return predicted.item(), confidence.item()


def handle_audio(path):
    try:
        # ... (audio loading and preprocessing is the same) ...
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE: waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.shape[1] < MAX_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, MAX_LEN - waveform.shape[1]))
        else:
            waveform = waveform[:, :MAX_LEN]
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024,
                                                             hop_length=512)
        mel_spec = mel_transform(waveform).unsqueeze(0).to(device)

        with torch.no_grad():
            output = audio_model(mel_spec)
            probabilities = F.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"Error processing audio: {e}")
        return -1, 0


def handle_video(path):
    try:
        # ... (video loading and preprocessing is the same) ...
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frames_tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

        with torch.no_grad():
            output = video_model(frames_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"Error processing video: {e}")
        return -1, 0


# === MODIFIED: Main Django View now handles the confidence score ===

def upload_file(request):
    form = MediaUploadForm()
    context = {'form': form, 'result': None}

    if request.method == 'POST':
        form = MediaUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            ext = os.path.splitext(uploaded_file.name)[1].lower()

            # Save file temporarily
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            prediction, confidence = -1, 0
            if ext in ['.jpg', '.jpeg', '.png']:
                prediction, confidence = handle_image(file_path)
            elif ext in ['.wav', '.mp3', '.flac']:
                prediction, confidence = handle_audio(file_path)
            elif ext in ['.mp4', '.avi', '.mov']:
                prediction, confidence = handle_video(file_path)
            else:
                context['result'] = {'text': 'Unsupported file type.', 'class': 'error'}

            # Format the result for the template
            confidence_percent = f"{confidence * 100:.2f}%"
            if prediction == 1:
                context['result'] = {
                    'text': 'Warning: This media is likely a DEEPFAKE.',
                    'class': 'fake',
                    'confidence': confidence_percent
                }
            elif prediction == 0:
                context['result'] = {
                    'text': 'This media appears to be REAL.',
                    'class': 'real',
                    'confidence': confidence_percent
                }
            elif prediction == -1:
                context['result'] = {'text': 'Could not process the file.', 'class': 'error'}

            os.remove(file_path)

    return render(request, 'predictor/upload.html', context)