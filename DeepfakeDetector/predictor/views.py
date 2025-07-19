from django.shortcuts import render
from .forms import MediaUploadForm
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
from django.conf import settings


image_model = resnet50(weights=None)
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2)

state_dict = torch.load(
    r'D:\Projects\PythonProject\DeepFake_Detector\Models\Image\deepfake_image_model1_stateDict.pth',
    map_location='cpu'
)
image_model.load_state_dict(state_dict)
image_model.eval()

# TODO: load audio and video models too...


image_model = resnet50(weights=None)  # safe, no ImageNet weights
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2)

state_dict = torch.load(
    r'D:\Projects\PythonProject\DeepFake_Detector\Models\Image\deepfake_image_model1_stateDict.pth',
    map_location='cpu'
)
image_model.load_state_dict(state_dict)
image_model.eval()

# TODO: load audio and video models too...

def handle_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(img).unsqueeze(0)
    output = image_model(img)
    _, predicted = torch.max(output, 1)
    return 'Fake' if predicted.item() == 1 else 'Real'

def handle_audio(path):
    # TODO: Add your audio model & preprocessing here
    return 'Audio detection not implemented'

def handle_video(path):
    # TODO: Add your video model & preprocessing here
    return 'Video detection not implemented'

def upload_file(request):
    result = None
    if request.method == 'POST':
        form = MediaUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            ext = os.path.splitext(uploaded_file.name)[1].lower()

            # ✅ Use BASE_DIR to get absolute temp folder
            temp_dir = os.path.join(settings.BASE_DIR, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            if ext in ['.jpg', '.jpeg', '.png']:
                result = handle_image(file_path)
            elif ext in ['.wav', '.mp3']:
                result = handle_audio(file_path)
            elif ext in ['.mp4', '.avi', '.mov']:
                result = handle_video(file_path)
            else:
                result = 'Unsupported file type'

            os.remove(file_path)

    else:
        form = MediaUploadForm()

    return render(request, 'predictor/upload.html', {'form': form, 'result': result})
