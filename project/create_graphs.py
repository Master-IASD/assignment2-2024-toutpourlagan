import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def plot_images(image_path, number, save_path):
    # Get all image file paths from the specified folder
    image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    
    # Check if there are enough images
    if number > len(image_files):
        raise ValueError("Number of images requested exceeds the number of images in the folder.")

    # Randomly select the specified number of images
    selected_images = random.sample(image_files, number)
    
    # Set the number of rows and columns to achieve three rows
    rows = 3
    cols = (number + 2) // 3  # Calculate the columns needed for three rows, rounding up
    
    # Create a figure to plot the images
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each selected image
    for ax, img_path in zip(axes, selected_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Hide any remaining subplots if the grid is larger than the number of images
    for ax in axes[number:]:
        ax.axis('off')

    # Add a legend/title
    fig.suptitle("Generated Images", fontsize=16)

    # Save the plot to the specified save path
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "samples_wganGP100.png")
    plt.savefig(save_file)
    plt.close()
    
    print(f"Plot saved as {save_file}")


if __name__ == '__main__':
    plot_images('generated_samples/WganGP', 21, 'plots')