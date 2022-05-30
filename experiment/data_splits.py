"""
Adapted from https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/experiment/splits.py.
"""
import random
from math import ceil
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

from utils.seeds import development_seed


def set_train_val_test_split_frac(seed: int, data: Data, val_frac: float, test_frac: float) -> Data:
    num_nodes = data.y.shape[0]

    val_size = ceil(val_frac * num_nodes)
    test_size = ceil(test_frac * num_nodes)
    train_size = num_nodes - val_size - test_size

    nodes = list(range(num_nodes))

    # Take same test set every time using development seed for robustness
    random.seed(development_seed)
    random.shuffle(nodes)
    test_idx = sorted(nodes[:test_size])
    nodes = [x for x in nodes if x not in test_idx]

    # Take train / val split according to seed
    random.seed(seed)
    random.shuffle(nodes)
    train_idx = sorted(nodes[:train_size])
    val_idx = sorted(nodes[train_size:])

    assert len(train_idx) + len(val_idx) + len(test_idx) == num_nodes

    data.train_mask = get_mask(train_idx, num_nodes)
    data.val_mask = get_mask(val_idx, num_nodes)
    data.test_mask = get_mask(test_idx, num_nodes)

    return data


def set_train_val_test_split(seed: int, data: Data, num_development: int = 1500, num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, min(num_per_class, int(len(class_idx) * 0.7)), replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    data.train_mask = get_mask(train_idx, num_nodes)
    data.val_mask = get_mask(val_idx, num_nodes)
    data.test_mask = get_mask(test_idx, num_nodes)

    return data


def get_mask(idx: List, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask
