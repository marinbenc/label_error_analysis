import random
import os
import os.path as p
import shutil
import json
import sys
from typing import List, Tuple, Dict, Callable, Optional, Literal

import fire
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from monai.losses.dice import DiceLoss

from functools import partial

import data.datasets as data
import utils
import models.models as models
from trainer import Trainer

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

DatasetType = Literal['seg_isic']

def get_model(
    log_dir: str, 
    dataset: DatasetType, 
    device: str, 
    fold: int,
    pretrained_model=None,) -> nn.Module:

    checkpoint = None
    if pretrained_model is not None:
        pretrained_dir = p.join('runs', pretrained_model, f'fold0')
        # take the 0th fold for pre-training
        checkpoint = p.join(pretrained_dir, f'best_fold=0.pth')
        print('Loading checkpoint from:', checkpoint)
    
    model = models.get_segmentation_model(dataset, device, checkpoint)

    return model

def train(*, 
    batch_size: int = 64, 
    epochs: int = 100, 
    lr: float = 0.001, 
    dataset: DatasetType,
    log_name: str, 
    device: str = 'cuda',
    folds: int = 1,
    overwrite: bool = False,
    workers: int = 0,
    pretrained_model: Optional[str] = None, label_error_percent=0.0, bias=0):
    """
    Train a detection or segmentation model. 
    
    The model checkpoints are saved in runs/log-name/model-type. 

    Args:
        dataset ('seg_isic'): The dataset to train on.
        folds (int): The number of folds to train on. If folds > 1, then the model is trained on a k-fold cross validation.
        log_name (str): The name of the log directory. The model checkpoints are saved in runs/log-name/model-type.
        device (str): The device to train on.
        overwrite (bool): If True, the log_name directory is deleted before training.
        workers (int): The number of workers to use for data loading.
        pretrained_model (str): The log_name of a pretrained model to initialize the model with.
    """
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)
    os.makedirs(name=f'runs/{log_name}', exist_ok=True)

    datasets = []

    dataset_args_valid = {
        'subset': 'valid',
        'augment': True,
        'colorspace': 'rgb',
        # The validation dataset also includes label errors, 
        # since in a real-world scenario, the model would be 
        # trained on data with label errors.
        'label_error_percent': label_error_percent,
        'bias': bias
    }

    dataset_args_train = {
        **dataset_args_valid,
        'subset': 'train',
        'augment': True,
        'label_error_percent': label_error_percent,
        'bias': bias
    }

    dataset_class = data.get_dataset_class(dataset)

    if folds <= 1:
        train_dataset = dataset_class(**dataset_args_train)
        valid_dataset = dataset_class(**dataset_args_valid)
        datasets.append((train_dataset, valid_dataset))
        json_dict = {
            'train_subjects': [list(train_dataset.subjects)],
            'valid_subjects': [list(valid_dataset.subjects)]
        }
        with open(f'runs/{log_name}/subjects.json', 'w') as f:
            json.dump(json_dict, f)
    else:
        whole_args_valid = dataset_args_valid.copy()
        whole_args_valid['subset'] = 'all'
        whole_args_train = dataset_args_train.copy()
        whole_args_train['subset'] = 'all'
        whole_dataset = dataset_class(**whole_args_valid)
        subject_ids = list(whole_dataset.subjects)
        subject_ids = sorted(subject_ids)

        existing_split = p.join('runs', log_name, 'subjects.json')
        if p.exists(existing_split):
            print('Using existing subject split')
            with open(existing_split, 'r') as f:
                json_dict = json.load(f)
                splits = zip(json_dict['train_subjects'], json_dict['valid_subjects'])
        else:
            kfold = KFold(n_splits=folds, shuffle=True, random_state=2022)
            splits = list(kfold.split(subject_ids))
            # convert from indices to subject ids
            splits = [([subject_ids[idx] for idx in train_idx], [subject_ids[idx] for idx in valid_idx]) for train_idx, valid_idx in splits]
            json_dict = {
                'train_subjects': [ids for (ids, _) in splits],
                'valid_subjects': [ids for (_, ids) in splits]
            }
            with open(f'runs/{log_name}/subjects.json', 'w') as f:
                json.dump(json_dict, f)

        for fold, (train_ids, valid_ids) in enumerate(splits):
            train_dataset = dataset_class(**whole_args_train, subjects=train_ids)
            valid_dataset = dataset_class(**whole_args_valid, subjects=valid_ids)
            # check for data leakage
            intersection = set(train_dataset.subjects).intersection(set(valid_dataset.subjects))
            assert len(intersection) == 0, f'Found {len(intersection)} overlapping subjects in fold {fold}'
            datasets.append((train_dataset, valid_dataset))
    
    os.makedirs(name=f'runs/{log_name}', exist_ok=True)
    if command_string and command_string is not None:
        command_file = p.join('runs', log_name, 'command.sh')
        with open(command_file, 'w') as f:
            f.write(command_string)

    for fold, (train_dataset, valid_dataset) in enumerate(datasets):
        print('----------------------------------------')
        print(f'Fold {fold}')
        print('----------------------------------------')

        log_dir = f'runs/{log_name}/fold{fold}'
        if p.exists(log_dir):
            if overwrite:
                shutil.rmtree(log_dir)
            else:
                raise ValueError(f'Log directory already exists: {log_dir}. Use --overwrite to overwrite.')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init, num_workers=workers)
        valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init, num_workers=workers)

        dice_loss = DiceLoss()
        def loss(pred, target):
            target_seg = target['seg']
            return dice_loss(pred, target_seg)
        validation_fn = loss

        model = get_model(log_dir, train_dataset, device, fold, pretrained_model)
        # model = torch.compile(model) # TODO: Figure out torch saving with compiled models
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True, min_lr=1e-15, eps=1e-15)
        
        trainer = Trainer(model, optimizer, loss, train_loader, valid_loader, validation_fn=validation_fn,
                          log_dir=log_dir, checkpoint_name=f'best_fold={fold}.pth', scheduler=scheduler)
        trainer.train(epochs)

if __name__ == '__main__':
    command_string = ' '.join(sys.argv)
    fire.Fire(train)