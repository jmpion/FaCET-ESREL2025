import numpy as np
import random
import torch

def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)