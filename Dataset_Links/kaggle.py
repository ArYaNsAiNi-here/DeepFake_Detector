import kagglehub

# Download latest version
path = kagglehub.dataset_download("abdallamohamed312/in-the-wild-audio-deepfake")
path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")

print("Path to dataset files:", path)