import pdb
import os
import torch
import numpy as np
import random
from PIL import ImageFilter
import torch.distributed as dist
import json
from .samplers import SubsetRandomSampler
from data.deepfake import Deepfake as deepfake_dataset

def build_loader(config, train=False, epoch=None):

    dataset_train = deepfake_dataset(phase='train', csv_dir='default')

    if train:
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
      
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.BATCH_SIZE_TRAIN,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True,
            shuffle=False,
            collate_fn=lambda x:dataset_train.collate_fn(x,'train'),
            )
        
        return dataset_train,data_loader_train

    else:
        dataset_val =deepfake_dataset(phase='val', csv_dir='default')
        dataset_test = deepfake_dataset(phase='test', csv_dir='default')

        indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)

        indices = np.arange(dist.get_rank(), len(dataset_test), dist.get_world_size())
        sampler_test = SubsetRandomSampler(indices)



        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=False,
            collate_fn=lambda x:dataset_val.collate_fn(x, 'val')
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=config.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=False,
            collate_fn=lambda x:dataset_test.collate_fn(x, 'val')
        )

        return dataset_train, data_loader_val, data_loader_test


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

