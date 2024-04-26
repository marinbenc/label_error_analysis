import os.path as p
from typing import List, Tuple, Dict, Callable, Optional, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

import preprocessing_utils as pre

import os

from label_error_sim import make_error_label, get_simplified_label, get_max_distance

def LesionSegDatasetISIC(subset, **kwargs):
  return LesionSegmentationDataset(subset=subset, dataset_folder='isic', **kwargs)

class LesionSegmentationDataset(torch.utils.data.Dataset):
  """
  A dataset for segmenting skin lesions.

  Args:
    subset ('train', 'valid', 'test' or 'all'): The subset of the dataset to use. Can be 'train', 'valid', 'test', or 'all'.
    dataset_folder (str): The folder containing the images and labels. The directory must be in the same directory as this file. 
    Expected folder structure:
      dataset_folder
      ├── train
      │   ├── input
      │   │   ├── <subject_name 0>.png
      │   │   ├── <subject_name 1>.png
      │   │   └── ...
      │   └── label
      │       ├── <subject_name 0>.png
      │       ├── <subject_name 1>.png
      │       └── ...
      ├── valid
      │   ├── input
      │   │   ├── ...
      │   └── label
      │       ├── ...
      ├── test
      │   ├── input
      │   │   ├── ...
      │   └── label
      │       ├── ...
    subjects (List[str]): 
      A list of subject names to use. If None, all subjects are used.
    augment (bool): 
      Whether to augment the dataset. If colorspace is 'dist', the image will be tinted with a randomly sampled color.
    colorspace ('lab', 'rgb'): 
      The colorspace to use.

    Attributes:
      subjects (set[str]): Names of the subjects in the dataset.
      num_classes (int): Number of classes in the dataset.
      subject_id_for_idx (List[str]): The subject name for each image in the dataset.
      file_names (List[str]): The file name for each image in the dataset.
  """
  def __init__(self, 
               subset: Literal['train', 'valid', 'test', 'all'], 
               dataset_folder: str, 
               subjects: Optional[List[str]] = None, 
               augment = False, 
               colorspace: Literal['lab', 'rgb']='rgb',
               label_error_percent=0.0, 
               ratio=1.0):
    self.dataset_folder = dataset_folder
    self.colorspace = colorspace
    self.num_classes = 3
    self.augment = augment
    self.label_error_percent = label_error_percent
    self.ratio = ratio

    assert self.colorspace in ['lab', 'rgb']

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    self.file_names = self._get_files(directories)
    if subjects is not None:
      self.file_names = [f for f in self.file_names if self._get_subject_from_file_name(f) in subjects]
    
    self.subject_id_for_idx = [self._get_subject_from_file_name(f) for f in self.file_names]
    self.subjects = subjects if subjects is not None else set(self.subject_id_for_idx)
    
  def _get_files(self, directories):
    file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, directory)
      directory_files = os.listdir(p.join(directory, 'label'))
      directory_files = [p.join(directory, 'label', f) for f in directory_files]
      directory_files.sort()
      file_names += directory_files
      file_names.sort()
    return file_names

  def _get_subject_from_file_name(self, file_name):
    return file_name.split('/')[-1].split('.')[0]
  
  def get_train_augmentation(self):
    return A.Compose([
      A.Flip(p=0.4),
      A.ShiftScaleRotate(p=0.4, rotate_limit=90, scale_limit=0.1, shift_limit=0.1, border_mode=cv.BORDER_CONSTANT, value=0, rotate_method='ellipse'),
      A.GridDistortion(p=0.4, border_mode=cv.BORDER_CONSTANT, value=0)
    ])
  
  def __len__(self):
    return len(self.file_names)
  
  def get_item_np(self, idx, augmentation=None):
    current_file = self.file_names[idx]
    input_path = current_file.replace('label/', 'input/').replace('.png', '.jpg')

    input = cv.imread(input_path)
    
    if self.colorspace == 'lab':
      input = cv.cvtColor(input, cv.COLOR_BGR2LAB)
    elif self.colorspace == 'rgb':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    
    input = input.transpose(2, 0, 1)

    mask = cv.imread(current_file, cv.IMREAD_GRAYSCALE)
    
    # Simulate label error
    if self.label_error_percent > 0:
      # TODO: This could be done more efficiently by only calculating the max_dist once per image
      # in the __init__ method and storing it in an array
      simplified_label = get_simplified_label(mask, 0)
      # Calculate a maximum distance between the simplified label and the original label
      # to make sure the errors are relative to the size of the object
      error = False
      max_dist = get_max_distance(mask, simplified_label)
      if max_dist == 0:
        print(f'Error for file {current_file}: max_dist is 0')
      else:
        simplified_label = get_simplified_label(mask, int(self.ratio * max_dist))
        try:
          mask = make_error_label(mask, simplified_label, self.label_error_percent)
        except Exception as e:
          print(f'Error for file {current_file}: {e}')
    
    mask = mask.astype(np.float32)
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    if augmentation is not None:
      input = input.transpose(1, 2, 0)
      transformed = augmentation(image=input, mask=mask)
      input = transformed['image']
      input = input.transpose(2, 0, 1)
      mask = transformed['mask']

    mask = mask.astype(np.float32)
    return input, mask

  def __getitem__(self, idx):
    input, label = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    to_tensor = ToTensorV2()
    input = input.astype(np.float32)
    input = input / 255.
    
    input_tensor, label_tensor = to_tensor(image=input.transpose(1, 2, 0), mask=label).values()
    input_tensor = input_tensor.float()
    label_tensor = label_tensor.unsqueeze(0).float()

    # visualize contour on input
    # viz = input.transpose(1, 2, 0)
    # viz = (viz * 255).astype(np.uint8)
    # contour = cv.findContours(label.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(viz, contour[0], -1, (0, 255, 0), 1)
    # plt.imshow(viz)
    # plt.show()

    return input_tensor, {'seg': label_tensor}
