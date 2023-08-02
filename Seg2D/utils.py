# This file is just to check if torch initializes cudnn properly
# If this raise CUDANN_STATUS_NOT_INITIALIZED
# Then delete the virtual environment and create a new one
# Then install pytorch again
import torch

def force_cudnn_initialization():
    s = 8
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))