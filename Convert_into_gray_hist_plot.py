import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the environment variable to avoid the Qt platform plugin error
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Directory containing input images
input_dir = ""
# Directory to save grayscale and brightened images
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# Brightness increase value
brightness_increase = 100

# Function to process an image: convert to grayscale, brighten, adjust contrast, and save
def process_image(input_path, output_image_path, output_histogram_path):
    # Read the image
    image = cv2.imread(input_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {input_path}")
        return
    
    # Resize the image to 512x512 pixels
    resized_image = cv2.resize(image, (512, 512))

    # Convert the resized image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Brighten the image by adding a constant to each pixel
    brightened_image = cv2.add(gray_image, brightness_increase)
    brightened_image = np.clip(brightened_image, 0, 255)  # Clip values to the range [0, 255]

    # Adjust contrast
    min_intensity = np.min(brightened_image)
    max_intensity = np.max(brightened_image)
    contrast_adjusted_image = (brightened_image - min_intensity) / (max_intensity - min_intensity) * 255
    contrast_adjusted_image = contrast_adjusted_image.astype(np.uint8)

    # Save the contrast-adjusted image
    cv2.imwrite(output_image_path, contrast_adjusted_image)

    # Calculate and display the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(contrast_adjusted_image.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram of Contrast-Adjusted Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Save the histogram figure
    dpi = 300
    plt.savefig(output_histogram_path, format='jpeg', dpi=dpi)
    plt.close()
    print(f"Processed and saved image and histogram: {output_image_path}, {output_histogram_path}")

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)
        output_histogram_path = os.path.join(output_dir, filename.replace('.png', '_hist.jpg').replace('.jpg', '_hist.jpg').replace('.jpeg', '_hist.jpg'))
        process_image(input_path, output_image_path, output_histogram_path)
