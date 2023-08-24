import torch
import torchvision

from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

def create_Unet(in_channels=3, num_classes=2):
    model = smp.Unet(
            encoder_name="resnet101",       
            encoder_weights="imagenet",    
            in_channels=in_channels,                  
            classes=num_classes,                      
    )
    
    params = get_preprocessing_params('resnet101', pretrained='imagenet')
    transform  = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=params['mean'], std=params['std']),
    ])
        
    return model, transform

def load_model_torch(model, path):
    """
    Load a PyTorch model with its state dictionary from a pth file.

    Args:
        model (torch.nn.Module): An instance of a PyTorch model where the state dictionary will be loaded.
        path (str): The file path from which the model state dictionary will be loaded.

    Returns:
        torch.nn.Module: The PyTorch model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(path), strict=False)   # Load Model
    return model