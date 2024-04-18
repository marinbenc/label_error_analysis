from typing import List, Union
from functools import partial
import pandas as pd
import numpy as np

from data.segmentation_dataset import LesionSegDatasetISIC
from segmentation_dataset import LesionSegmentationDataset


dataset_to_class = {
    'seg_isic': LesionSegDatasetISIC,
}

dataset_choices = dataset_to_class.keys()

def get_dataset_class(dataset_name: Union[str, List[str]]):
  if isinstance(dataset_name, list):
    return partial(composed_dataset, dataset_names=dataset_name)

  if dataset_name not in dataset_to_class:
    raise ValueError(f'Unknown dataset {dataset_name}')
  return dataset_to_class[dataset_name]

def composed_dataset(dataset_names: List[str], **kwargs):
  datasets = [get_dataset_class(name)(**kwargs) for name in dataset_names]
  dataset1 = datasets[0]
  for dataset2 in datasets[1:]:
    if isinstance(dataset1, LesionSegmentationDataset):
      dataset1.skin_colors_df = pd.concat([dataset1.skin_colors_df, dataset2.skin_colors_df])
      dataset1.subjects = set(list(dataset1.subjects) + list(dataset2.subjects))
      dataset1.skin_colors = dataset1.skin_colors + dataset2.skin_colors
      dataset1.file_names = dataset1.file_names + dataset2.file_names
      dataset1.subject_id_for_idx += dataset2.subject_id_for_idx
  return dataset1