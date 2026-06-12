import os
import random
import numpy as np
import torch
import albumentations as A


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

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except RuntimeError:
        # Starije verzije PyTorcha ne podrzavaju warn_only parametar
        torch.use_deterministic_algorithms(True)

    # Potrebno za deterministicke CUDA operacije (scatter, gather, i sl.)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"Seed set to {seed}")
    print(f"  - PYTHONHASHSEED: {os.environ['PYTHONHASHSEED']}")
    print(f"  - torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  - torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - torch.use_deterministic_algorithms: True")
    print(f"  - Albumentations seed: {seed}")
    print(f"  - CUBLAS_WORKSPACE_CONFIG: {os.environ['CUBLAS_WORKSPACE_CONFIG']}")