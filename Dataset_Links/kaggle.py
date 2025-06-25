# The following code will only execute
# successfully when compression is complete

import kagglehub

# Download latest version
path = kagglehub.dataset_download("abdallamohamed312/in-the-wild-audio-deepfake")

print("Path to dataset files:", path)