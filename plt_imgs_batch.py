import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import signal
from concurrent.futures import ProcessPoolExecutor

# Set paths to the directories
image_dir = 'data/isic/all_images'
ground_truth_dir = 'data/isic/all_labels_ground_truth'
base_distorted_dir = 'data/isic'
save_directory = 'plotted_imgs'

# Ensure save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define error percentages and specific order of biases
label_error_percent = ['25', '50', '75', '100']
biases = ['-1', '0', '1']

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def process_image(image_name):
    # Prepare the paths
    img_path = os.path.join(image_dir, image_name)
    ground_truth_path = os.path.join(ground_truth_dir, image_name.replace('.jpg', '.png'))

    # Load the image and ground truth
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Create a figure with 4 rows and 3 columns, each row for a different error percentage
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))

    for row_idx, percent in enumerate(label_error_percent):
        for col_idx, bias in enumerate(biases):
            distorted_label_path = os.path.join(base_distorted_dir, f'labels_{percent}_{bias}', image_name.replace('.jpg', '.png'))
            distorted_label = cv2.imread(distorted_label_path, cv2.IMREAD_GRAYSCALE)

            # Find contours for ground truth and distorted labels
            contours_gt, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_distorted, _ = cv2.findContours(distorted_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ax = axs[row_idx, col_idx]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'Error {percent}% Bias {bias}')

            for contour in contours_gt:
                ax.plot(contour[:, :, 0], contour[:, :, 1], color='limegreen', linestyle='--', linewidth=2)
            for contour in contours_distorted:
                ax.plot(contour[:, :, 0], contour[:, :, 1], color='magenta', linestyle='--', linewidth=2)

            if row_idx == 0 and col_idx == 0:
                ax.legend(['Ground Truth', 'Distorted'], loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(save_directory, f'{os.path.splitext(image_name)[0]}.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# Using ProcessPoolExecutor to utilize multiple cores
if __name__ == '__main__':
    # List of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU
        executor.map(process_image, image_files)