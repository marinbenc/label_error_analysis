# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Set paths to the directories
# image_dir = 'data/isic/all_images'
# ground_truth_dir = 'data/isic/all_labels_ground_truth'
# base_distorted_dir = 'data/isic'
# save_directory = 'plotted_imgs'

# # Ensure save directory exists
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)

# # Select the first image for demonstration
# image_name = next((f for f in os.listdir(image_dir) if f.endswith('.jpg')), None)

# # Define error percentages and specific order of biases
# label_error_percent = ['25', '50', '75', '100']
# biases = ['-1', '0', '1']

# # Create a figure with 4 rows and 3 columns, each row for a different error percentage
# fig, axs = plt.subplots(4, 3, figsize=(12, 16))  # Adjust the size as needed

# # Load the image and ground truth once since it's the same for all subplots
# img_path = os.path.join(image_dir, image_name)
# ground_truth_path = os.path.join(ground_truth_dir, image_name.replace('.jpg', '.png'))
# image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

# for row_idx, percent in enumerate(label_error_percent):
#     for col_idx, bias in enumerate(biases):
#         distorted_label_path = os.path.join(base_distorted_dir, f'labels_{percent}_{bias}', image_name.replace('.jpg', '.png'))
#         distorted_label = cv2.imread(distorted_label_path, cv2.IMREAD_GRAYSCALE)

#         # Find contours for ground truth and distorted labels
#         contours_gt, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours_distorted, _ = cv2.findContours(distorted_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         ax = axs[row_idx, col_idx]
#         ax.imshow(image)
#         ax.axis('off')
#         ax.set_title(f'Error {percent}% Bias {bias}')

#         # Draw contours with matplotlib
#         for contour in contours_gt:
#             ax.plot(contour[:, :, 0], contour[:, :, 1], 'g--', linewidth=2)  # Green dashed line for ground truth
#         for contour in contours_distorted:
#             ax.plot(contour[:, :, 0], contour[:, :, 1], 'r--', linewidth=2)  # Red dashed line for distorted labels

#         # Adding a custom legend
#         if row_idx == 0 and col_idx == 0:  # Add legend only in the first subplot to avoid repetition
#             ax.legend(['Ground Truth', 'Distorted'], loc='upper right')

# plt.tight_layout()
# save_path = os.path.join(save_directory, 'single_image_detailed_plots.png')
# plt.savefig(save_path)  # Save the figure with all plots
# plt.show()

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

from utils import fpr, fnr

#How to run the scrypt: python your_script_name.py image.jpg

# Check if image name is provided as a command line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <image_name>")
    sys.exit(1)

image_name = sys.argv[1]

# Set paths to the directories
image_dir = 'data/isic/test/input'
ground_truth_dir = 'data/isic/test/label'
base_distorted_dir = 'data/isic'
save_directory = 'plotted_imgs'

# Ensure save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define error percentages and specific order of biases
label_error_percent = ['25', '50', '75', '100']
biases = ['-1', '0', '1']

# Prepare the paths
img_path = os.path.join(image_dir, image_name)
ground_truth_path = os.path.join(ground_truth_dir, image_name.replace('.jpg', '.png'))

if not os.path.exists(img_path) or not os.path.exists(ground_truth_path):
    print("Image or label not found.")
    sys.exit(1)

# Load the image and ground truth
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

# Create a figure with 4 rows and 3 columns, each row for a different error percentage
fig, axs = plt.subplots(4, 3, figsize=(12, 16))  # Adjust the size as needed

for row_idx, percent in enumerate(label_error_percent):
    for col_idx, bias in enumerate(biases):
        distorted_label_path = os.path.join(base_distorted_dir, f'labels_{percent}_{bias}', image_name.replace('.jpg', '.png'))
        distorted_label = cv2.imread(distorted_label_path, cv2.IMREAD_GRAYSCALE)

        print(f'fpr: {fpr(distorted_label, ground_truth)}')
        print(f'fnr: {fnr(distorted_label, ground_truth)}')

        # Find contours for ground truth and distorted labels
        contours_gt, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_distorted, _ = cv2.findContours(distorted_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ax = axs[row_idx, col_idx]
        ax.imshow(ground_truth * 0.5 + 0.5 * distorted_label, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Error {percent}% Bias {bias}')

        print(f'row_idx: {row_idx}, col_idx: {col_idx}')
        print(f'bias: {bias}, percent: {percent}')

        # Draw contours with matplotlib
        for contour in contours_gt:
            ax.plot(contour[:, :, 0], contour[:, :, 1], color='limegreen', linestyle='--', linewidth=2)
        for contour in contours_distorted:
            ax.plot(contour[:, :, 0], contour[:, :, 1], color='magenta', linestyle='--', linewidth=2)

        # Adding a custom legend
        if row_idx == 0 and col_idx == 0:
            ax.legend(['Ground Truth', 'Distorted'], loc='upper right')

plt.tight_layout()
save_path = os.path.join(save_directory, f'{os.path.splitext(image_name)[0]}_detailed_plots.png')
plt.show()
#plt.savefig(save_path, dpi=300)  # Save the figure with all plots
#plt.close(fig)  # Close the figure to free up memory
