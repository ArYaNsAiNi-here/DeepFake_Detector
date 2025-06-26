import os
folder_path = 'D:\\Dev\\Code10Thrive\\2\\for-2sec\\for-2seconds\\training\\fake'

def get_all_extensions(folder_path):
    extensions = set()
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext:
                extensions.add(ext)
    return sorted(extensions)

# 🔍 Example usage
folder = "./audio_dataset"
exts = get_all_extensions(folder)
print("Found extensions:", exts)
