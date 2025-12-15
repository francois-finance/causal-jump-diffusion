import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    Fige l'aléatoire pour l'ensemble de la stack (Python, Numpy, PyTorch).
    Gère CPU, GPU (Cuda) et Mac (MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Support spécifique Mac M1/M2/M3
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔒 [Utils] Seed fixée globalement à : {seed}")