import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# === CONFIGURATION ===
dataset_dir = "D:\\Dev\\Code10Thrive\\image_dataset"  # Replace with your actual dataset path
output_csv = "deepfake_image_dataset.csv"

# Optional: Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# === METADATA EXTRACTION ===
def extract_metadata(image_path, label):
    info = {
        "file_path": image_path,
        "label": label,
        "width": None,
        "height": None,
        "file_size_kb": None,
        "face_detected": False,
        "avg_color": None,
    }

    try:
        img = cv2.imread(image_path)
        if img is None:
            return info  # Skip unreadable images

        # Basic details
        h, w = img.shape[:2]
        info["width"] = w
        info["height"] = h
        info["file_size_kb"] = os.path.getsize(image_path) / 1024
        info["avg_color"] = np.mean(img, axis=(0, 1)).tolist()

        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        info["face_detected"] = len(faces) > 0

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return info


# === PROCESS ALL IMAGES ===
records = []
for label_folder in ['real', 'fake']:
    folder_path = os.path.join(dataset_dir, label_folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in tqdm(os.listdir(folder_path), desc=f"Scanning {label_folder}"):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(folder_path, fname)
            metadata = extract_metadata(fpath, label_folder)
            records.append(metadata)

# === SAVE TO CSV ===
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"Metadata saved to {output_csv}")
