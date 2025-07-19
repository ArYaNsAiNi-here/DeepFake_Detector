import torch
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm

from PIL import Image
import torchvision.transforms as transforms
import cv2

# Load your trained model once
model = torch.load('./models/deepfake_image_model.pth', map_location=torch.device('cpu'))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            fs = FileSystemStorage()
            file_path = fs.save(uploaded_file.name, uploaded_file)
            file_url = fs.url(file_path)

            # Make prediction
            img = Image.open(fs.path(file_path)).convert('RGB')
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                prediction = torch.argmax(output, dim=1).item()

            label = 'REAL' if prediction == 0 else 'FAKE'

            return render(request, 'predictor/result.html', {
                'label': label,
                'file_url': file_url
            })

    else:
        form = UploadFileForm()
    return render(request, 'predictor/upload.html', {'form': form})
