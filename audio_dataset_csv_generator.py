import os
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pydub import AudioSegment

# === CONFIGURATION ===
audio_dataset_dir = "D:\\Dev\\Code10Thrive\\audio_dataset"  # 🔁 e.g. 'audio_dataset/'
output_csv = "deepfake_audio_metadata.csv"
output_converted_dir = "converted_wav_dataset"

os.makedirs(output_converted_dir, exist_ok=True)

supported_exts = ('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a')


# === CONVERT TO WAV IF NEEDED ===
def convert_to_wav_if_needed(filepath, output_dir):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == '.wav':
        return filepath  # Already WAV

    filename = os.path.splitext(os.path.basename(filepath))[0]
    new_path = os.path.join(output_dir, filename + ".wav")

    try:
        audio = AudioSegment.from_file(filepath)
        audio.export(new_path, format="wav")
        return new_path
    except Exception as e:
        print(f"❌ Failed to convert {filepath}: {e}")
        return None


# === FEATURE EXTRACTION ===
def extract_features(filepath, label):
    features = {
        "file_path": filepath,
        "label": label,
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "file_size_kb": None,
        "rms_energy": None,
        "zero_crossing_rate": None
    }

    try:
        y, sr = librosa.load(filepath, sr=None)
        features["sample_rate"] = sr
        features["duration_sec"] = librosa.get_duration(y=y, sr=sr)
        features["channels"] = 1  # librosa loads mono by default
        features["file_size_kb"] = os.path.getsize(filepath) / 1024
        features["rms_energy"] = float(np.mean(librosa.feature.rms(y=y)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Extract mean MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, val in enumerate(np.mean(mfccs, axis=1)):
            features[f"mfcc_{i + 1}"] = float(val)

    except Exception as e:
        print(f"Failed to extract from {filepath}: {e}")

    return features


# === PROCESS ENTIRE DATASET ===
records = []

for label in ['real', 'fake']:
    label_folder = os.path.join(audio_dataset_dir, label)
    if not os.path.isdir(label_folder):
        continue

    for fname in tqdm(os.listdir(label_folder), desc=f"Scanning {label}"):
        if fname.lower().endswith(supported_exts):
            original_path = os.path.join(label_folder, fname)

            wav_path = convert_to_wav_if_needed(original_path, output_converted_dir)
            if wav_path:
                metadata = extract_features(wav_path, label)
                records.append(metadata)

# === SAVE TO CSV ===
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"\n✅ Audio metadata saved to {output_csv}")