# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
from torch import nn
import torch.distributed as dist
import random
import numpy as np
from torch.utils.data import DataLoader

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data(task_id, args, scenario_train, scenario_val):
    dataset_train = scenario_train[task_id]
    dataset_val = scenario_val[task_id]
    dataset_val_entire = scenario_val[:task_id+1]

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    loader_val_entire = DataLoader(dataset_val_entire, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return loader_train, loader_val, loader_val_entire

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
