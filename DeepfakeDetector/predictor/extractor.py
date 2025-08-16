import os
import json
import sys

from PIL import Image, ExifTags
import mutagen  # for audio metadata
import cv2      # for video metadata

def extract_image_metadata(file_path):
    """Extract EXIF metadata from an image file."""
    metadata = {}
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                metadata[tag_name] = str(value)
    except Exception as e:
        metadata['error'] = str(e)
    return metadata

def extract_audio_metadata(file_path):
    """Extract metadata from audio file (MP3/WAV)."""
    metadata = {}
    try:
        audio = mutagen.File(file_path, easy=True)
        if audio:
            for key, value in audio.items():
                metadata[key] = str(value)
    except Exception as e:
        metadata['error'] = str(e)
    return metadata

def extract_video_metadata(file_path):
    """Extract basic metadata from video file using OpenCV."""
    metadata = {}
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"error": "Unable to open video file"}

        metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
        metadata["duration_sec"] = (
            metadata["frame_count"] / metadata["fps"]
            if metadata["fps"] > 0 else None
        )
        metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as e:
        metadata['error'] = str(e)
    return metadata

def extract_metadata(file_path):
    """Decide based on file extension which extractor to use."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        return extract_image_metadata(file_path)
    elif ext in [".mp3", ".wav", ".flac"]:
        return extract_audio_metadata(file_path)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return extract_video_metadata(file_path)
    else:
        return {"error": "Unsupported file type"}

if __name__ == "__main__":
    metadata = extract_metadata("C:\\Users\\Asus\\Pictures\\Camera Roll\\WIN_20250717_10_58_27_Pro.jpg")
    print(metadata)