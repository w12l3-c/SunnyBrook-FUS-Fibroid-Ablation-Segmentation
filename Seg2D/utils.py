# =============================================================================
# File Description:
# ------------------
# This file is just to check if torch initializes cudnn properly
#
# If this still raise CUDANN_STATUS_NOT_INITIALIZED
# Then delete the virtual environment and create a new one
# Then install pytorch again
# =============================================================================

import torch

def force_cudnn_initialization():
    """
    Force initialization of CuDNN for GPU computations using dummy tensors.

    This function initializes CuDNN for GPU computations by performing a convolution operation
    on dummy tensors. It is useful to ensure consistent behavior and performance of CuDNN in some cases.

    Note:
        This function should be called once at the beginning of your PyTorch script
        if you encounter issues related to CuDNN performance or reproducibility.

    Example:
        To initialize CuDNN, simply call this function in your script:

        >>> force_cudnn_initialization()
    """
    s = 8
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))