import numpy as np
import json
from glob import glob
import torch
from torch.nn.functional import softmax, sigmoid
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import os.path as p

from torch.utils.data import DataLoader

import pandas as pd
import fire

import utils
import data.datasets as data
import models.models as models

device = 'cuda'

def calculate_metrics(ys_pred, ys, metrics, subjects=None):
  '''
  Parameters:
    ys_pred: model-predicted segmentation masks
    ys: the GT segmentation masks
    metrics: a dictionary of type `{metric_name: metric_fn}` 
    where `metric_fn` is a function that takes `(y_pred, y)` and returns a float.
    subjects: a list of subject IDs, one for each element in ys_pred. If provided, the
    returned DataFrame will have a column with the subject IDs.

  Returns:
    A DataFrame with one column per metric and one row per image.
  '''
  metric_names, metric_fns = list(metrics.keys()), metrics.values()
  columns = metric_names + ['subject']
  df = pd.DataFrame(columns=columns)

  if subjects is None:
    subjects = ['none'] * len(ys_pred)

  df['subject'] = subjects
  df['subject'] = df['subject'].astype('category')
  df.set_index(keys='subject', inplace=True)
  for (metric_name, fn) in metrics.items():
    df[metric_name] = [fn(y_pred, y) for (y_pred, y) in zip(ys_pred, ys)]
  
  return df

def calculate(dataset, label_error_percent, bias):
    datasets = []
    dataset_args = {
      'subset': 'train',
      'augment': False,
      'colorspace': 'rgb'
    }

    no_error_dataset = data.get_dataset_class(dataset)(**dataset_args)
    dataset_args['label_error_percent'] = label_error_percent
    dataset_args['bias'] = bias
    error_dataset = data.get_dataset_class(dataset)(**dataset_args)
    
    metrics = {
        'dsc': utils.dsc,
        'fpr': utils.fpr,
        'fnr': utils.fnr,
    }

    ys = []
    ys_error = []

    # TODO: Use data loader to speed this up

    for i in range(len(no_error_dataset)):
        _, y = no_error_dataset[i]
        y = y['seg'].squeeze().detach().cpu().numpy()
        y = utils.thresh(y)
        ys.append(y)

        _, y = error_dataset[i]
        y = y['seg'].squeeze().detach().cpu().numpy()
        y = utils.thresh(y)
        ys_error.append(y)

    df = calculate_metrics(ys, ys_error, metrics)
    print(df.info())
    print(df.describe())

    plt.show()

if __name__ == '__main__':
    fire.Fire(calculate)