import os
import os.path as p
import shutil
import numpy as np

def check_data():
  isic_folder = 'data/isic'
  subfolders = ['train', 'valid', 'test']
  # for subfolder in subfolders:
  #   input_folder = p.join(isic_folder, subfolder, 'input')
  #   label_folder = p.join(isic_folder, subfolder, 'label')
  #   input_files = os.listdir(input_folder)
  #   label_files = os.listdir(label_folder)
  #   assert len(input_files) == len(label_files)
  #   for input_file, label_file in zip(input_files, label_files):
  #     assert input_file == label_file

  # Check that all files in train, valid, and test are unique
  train_files = os.listdir(p.join(isic_folder, 'train', 'label'))
  valid_files = os.listdir(p.join(isic_folder, 'valid', 'label'))
  test_files = os.listdir(p.join(isic_folder, 'test', 'label'))

  assert len(set(train_files).intersection(set(valid_files))) == 0
  assert len(set(train_files).intersection(set(test_files))) == 0
  assert len(set(valid_files).intersection(set(test_files))) == 0

  print('Data check passed')

if __name__ == '__main__':
  check_data()

