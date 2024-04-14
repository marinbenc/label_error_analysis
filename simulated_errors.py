import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from label_error_sim import make_error_label, get_simplified_label, get_max_distance

class LabelErrorDataset(Dataset):
    def __init__(self, image_paths, label_paths, label_error_percent=0.0, ratio=1.0, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.label_error_percent = label_error_percent
        self.ratio = ratio

    def __len__(self):
        return len(self.image_paths)

    def get_item_np(self, idx):
        # Load image and label as numpy arrays
        image = cv.imread(self.image_paths[idx])
        label = cv.imread(self.label_paths[idx], cv.IMREAD_GRAYSCALE)  # Load as grayscale

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        # Simulate label error
        if self.label_error_percent > 0:
            simplified_label = get_simplified_label(label, int(self.ratio * get_max_distance(label, label)))
            label = make_error_label(label, simplified_label, self.label_error_percent)

        return image, label

    def __getitem__(self, idx):
        image, label = self.get_item_np(idx)
        #Convert to tensors if using in a PyTorch context, otherwise return numpy arrays directly
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to CxHxW
        label = torch.tensor(label)
        return image, label