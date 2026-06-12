import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set all random seeds for complete reproducibility.
    """
    # Python hashseed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    # PyTorch CUDA random (for GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior on cuDNN (slows down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Seed set to {seed}")
    print(f"  - PYTHONHASHSEED: {os.environ['PYTHONHASHSEED']}")
    print(f"  - torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  - torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
