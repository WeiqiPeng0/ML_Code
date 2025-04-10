import numpy as np
import torch

def softmax(x, axis=-1):
    """
    Compute softmax along a specified axis.
    
    Parameters:
    - x: np.ndarray, input array
    - axis: int, axis to perform softmax on (default: last axis)
    
    Returns:
    - softmaxed: np.ndarray, same shape as x
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


import torch

def softmax(x, axis=-1):
    """
    Manual implementation of softmax in PyTorch.
    
    Args:
        x (torch.Tensor): input tensor
        axis (int): axis to apply softmax over
    
    Returns:
        torch.Tensor: softmax result with same shape as input
    """
    # For numerical stability, subtract max
    x_max = torch.max(x, dim=axis, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=axis, keepdim=True)
    return e_x / sum_e_x


