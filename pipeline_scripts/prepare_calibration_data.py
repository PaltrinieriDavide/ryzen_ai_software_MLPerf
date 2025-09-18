import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import argparse # NUOVO

# La configurazione è stata rimossa, ora viene passata tramite argomenti
NUM_CALIB_SAMPLES = 320
IMAGE_SIZE = 224

# Pre-elaborazione
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MODIFICATO: La funzione ora accetta i percorsi come parametri
def prepare_data(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Error: Calibration image directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Error: No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} images. Using up to {NUM_CALIB_SAMPLES} for calibration.")

    count = 0
    for img_path in tqdm(image_files[:NUM_CALIB_SAMPLES], desc="Preprocessing calibration images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            img_numpy = img_tensor.unsqueeze(0).numpy()

            output_filename = os.path.join(output_dir, f"calib_input_{count}.npy")
            np.save(output_filename, img_numpy)

            count += 1
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")

    if count > 0:
        print(f"Successfully preprocessed {count} images into '{output_dir}'.")
    else:
        print("No images were preprocessed.")

# NUOVO: Blocco main per gestire gli argomenti della riga di comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess calibration images.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with raw JPEG/PNG calibration images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed .npy files.')
    args = parser.parse_args()
    
    prepare_data(args.input_dir, args.output_dir)