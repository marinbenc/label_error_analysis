import os
import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

from data.segmentation_dataset import LesionSegmentationDataset
from label_error_sim import make_error_label

def tensor_to_numpy(image_tensor, mask_tensor):
    # Ensure the tensors are moved to CPU memory if they are on a GPU
    image_np = image_tensor.cpu().numpy()
    mask_np = mask_tensor.cpu().numpy()

    # Assuming the input image was normalized to [0, 1] range before converted to a tensor
    image_np = image_np * 255  # Scale the image to [0, 255]
    image_np = image_np.astype(np.uint8)  # Convert to uint8 type

    # Transpose the image array if it's in (C, H, W) format to (H, W, C) for OpenCV compatibility
    if image_np.shape[0] < 4:  # Checking to avoid dimension issues if already in (H, W, C)
        image_np = np.transpose(image_np, (1, 2, 0))

    # For the mask, ensure it is a single channel (H, W), scale and convert similarly
    mask_np = mask_np.squeeze()  # Remove extra dimensions, expect (1, H, W) -> (H, W)
    mask_np = mask_np * 255
    mask_np = mask_np.astype(np.uint8)

    return image_np, mask_np

def get_error_labels(dataset, label_error_percent, bias):
    ys_error = []
    n = len(dataset)

    print(f"Dataset lenght: {n}")

    for i in range(n):
       image_tensor, mask_dict = dataset[i]  # Get tensor output from dataset
       mask_tensor = mask_dict['seg']
        
        # Convert tensors to NumPy arrays
       _, mask_np = tensor_to_numpy(image_tensor, mask_tensor)

       error_mask = make_error_label(mask_np, label_error_percent, bias)
       ys_error.append(error_mask)
    
    return ys_error

def save_error_labels(dataset, ys_error, output_folder):
    for i, error_mask in enumerate(ys_error):
        subject_id = dataset.subject_id_for_idx[i]
        output_path = os.path.join(output_folder, f"{subject_id}.png")
        cv.imwrite(output_path, error_mask)

# def main():
#     parser = argparse.ArgumentParser(description='Generate and save error masks for a dataset.')
#     parser.add_argument('--label_error_percent', type=float, required=True, help='Percentage of label error to apply, from 0.0 to 1.0.')
#     parser.add_argument('--bias', type=int, required=True, help='Bias towards false positives (1), false negatives (-1), or no bias (0).')
#     args = parser.parse_args()

#     # Initialize dataset
#     dataset = LesionSegmentationDataset(subset='all', dataset_folder='isic', augment=False, colorspace='rgb')

#     # Generate error labels
#     ys_error = get_error_labels(dataset, args.label_error_percent, args.bias)

#     # Define output folder and create it if not exists
#     output_folder = f"data/isic/labels_{args.label_error_percent}_{args.bias}"
#     Path(output_folder).mkdir(parents=True, exist_ok=True)

#     # Save error masks
#     save_error_labels(dataset, ys_error, output_folder)

def batch_process():
    biases = [-1, 0, 1]
    label_error_percents = np.arange(0.1, 1.1, 0.1)
    dataset = LesionSegmentationDataset(subset='all', dataset_folder='isic', augment=False, colorspace='rgb')

    for bias in biases:
        for label_error_percent in label_error_percents:
            print(f"Processing bias {bias}, error percent {label_error_percent:.2f}")
            ys_error = get_error_labels(dataset, label_error_percent, bias)
            output_folder = f"data/isic/labels_{int(label_error_percent * 100)}_{bias}"
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            save_error_labels(dataset, ys_error, output_folder)

if __name__ == '__main__':
    # main()
    batch_process()
