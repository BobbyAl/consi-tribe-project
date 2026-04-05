from PIL import Image
import os

# 1. Setup folders
input_dir = "brain_responses_1hz"  # Your original folder
output_dir = "brain_responses_cropped"
os.makedirs(output_dir, exist_ok=True)

# 2. Define your crop margins (in pixels) to remove labels
LEFT_CROP = 35    # Removes the "brain response" text
BOTTOM_CROP = 25  # Removes the "t=0" text
TOP_CROP = 5      # Optional: clean up the top edge
RIGHT_CROP = 5    # Optional: clean up the right edge

print("Starting the batch crop and resize...")

# 3. Processing Loop
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

# First, crop all images with fixed margins
cropped_images = {}
for filename in files:
    img = Image.open(os.path.join(input_dir, filename))
    width, height = img.size
    
    # Define the box: (left, top, right, bottom)
    box = (LEFT_CROP, TOP_CROP, width - RIGHT_CROP, height - BOTTOM_CROP)
    
    cropped_img = img.crop(box)
    cropped_images[filename] = cropped_img
    img.close()

# Get the size of the cropped response_053.png
target_filename = "response_053.png"
if target_filename in cropped_images:
    target_size = cropped_images[target_filename].size
    print(f"Target size from {target_filename}: {target_size}")
else:
    print(f"{target_filename} not found, using first cropped size")
    target_size = list(cropped_images.values())[0].size

# Resize all cropped images to the target size
for filename, cropped_img in cropped_images.items():
    resized_img = cropped_img.resize(target_size, Image.Resampling.LANCZOS)
    resized_img.save(os.path.join(output_dir, filename))

print(f"Done! {len(files)} images cropped, resized to {target_size}, and saved to /{output_dir}")