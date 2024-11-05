import os
from torchvision import datasets
from PIL import Image

# Load the test dataset without any transformations
test_dataset = datasets.MNIST(root='data/MNIST/', train=True, download=True)

# Create directory to save images
output_dir = 'test_samples'
os.makedirs(output_dir, exist_ok=True)

# Number of images to save
n_samples = 1000

# Loop through the first n_samples images and save each one
for i in range(n_samples):
    image, label = test_dataset[i]

    # Save the raw PIL image directly
    image.save(os.path.join(output_dir, f'image_{i}_label_{label}.png'))

print(f"{n_samples} images saved to '{output_dir}' directory.")