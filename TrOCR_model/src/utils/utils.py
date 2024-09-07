import gc

import torch


def clear_cuda_cache():
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')