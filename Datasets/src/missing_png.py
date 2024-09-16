import os
import shutil

# Define the paths
missing_images_file = './missing.txt'  # Replace with the actual path to your missing.txt file
source_dir = '/home/pool/LLM_OCR_DEMO_1/LLM_OCR/resources/Bentham/BenthamDatasetR0-GT/Images/Lines'
target_dir = '/home/pool/HTR_PostProcesing_LLM/Datasets/resources/BenthamDatasetR0-GT/Images/Lines'

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Read the missing image filenames
with open(missing_images_file, 'r') as f:
    missing_images = [line.strip() + '.png' for line in f if line.strip()]

# Copy the missing images
for image_name in missing_images:
    source_path = os.path.join(source_dir, image_name)
    target_path = os.path.join(target_dir, image_name)

    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        print(f"Copied {image_name} to {target_dir}")
    else:
        print(f"Image {image_name} not found in source directory.")

print("Copying process completed.")
