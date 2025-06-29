import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm # For progress bar
import os

# --- Configuration ---
CALIB_IMAGE_DIR = "small_imagenet"  # Directory with your raw JPEG calibration images
OUTPUT_CALIB_DATA_DIR = "calib_data_imagenet" # Directory to save preprocessed numpy arrays
NUM_CALIB_SAMPLES = 320
IMAGE_SIZE = 224

# --- Preprocessing ---
# Standard ImageNet normalization
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def prepare_data():
    if not os.path.exists(CALIB_IMAGE_DIR):
        print(f"Error: Calibration image directory '{CALIB_IMAGE_DIR}' not found.")
        print("Please create it and place your JPEG images there.")
        return

    os.makedirs(OUTPUT_CALIB_DATA_DIR, exist_ok=True)

    image_files = [os.path.join(CALIB_IMAGE_DIR, f)
                   for f in os.listdir(CALIB_IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Error: No images found in '{CALIB_IMAGE_DIR}'.")
        return

    print(f"Found {len(image_files)} images. Using up to {NUM_CALIB_SAMPLES} for calibration.")

    count = 0
    for img_path in tqdm(image_files[:NUM_CALIB_SAMPLES], desc="Preprocessing calibration images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            img_numpy = img_tensor.unsqueeze(0).numpy() # Add batch dimension and convert to numpy

            output_filename = os.path.join(OUTPUT_CALIB_DATA_DIR, f"calib_input_{count}.npy")
            np.save(output_filename, img_numpy)

            count += 1
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")

    if count > 0:
        print(f"Successfully preprocessed {count} images into '{OUTPUT_CALIB_DATA_DIR}'.")
        print("Each .npy file contains one preprocessed image tensor (batch_size=1).")
    else:
        print("No images were preprocessed.")


if __name__ == "__main__":
    prepare_data()