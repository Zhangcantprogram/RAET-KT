import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import random as python_random
# from pykt.datasets.data_loader import KTDataset

def set_seed(seed):
    """Set global RNGs and require deterministic PyTorch execution.
    
    Args:
        seed (int): random seed
    """
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
    # Raise on an operation for which PyTorch cannot provide a deterministic
    # implementation instead of silently producing an irreproducible run.
    torch.use_deterministic_algorithms(True)

    np.random.seed(seed)
    python_random.seed(seed)

import datetime
def get_now_time():
    """Return the time string, the format is %Y-%m-%d %H:%M:%S

    Returns:
        str: now time
    """
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string

def debug_print(text,fuc_name=""):
    """Printing text with function name.

    Args:
        text (str): the text will print
        fuc_name (str, optional): _description_. Defaults to "".
    """
    print(f"{get_now_time()} - {fuc_name} - said: {text}")
