"""
Reproducibility utilities for setting random seeds.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")


def worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader to ensure reproducibility.

    Args:
        worker_id: Worker ID from DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
