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

def accuracy_dice(pred, target):
    """
    Calculate Dice coefficient accuracy between predicted and target binary masks.

    Args:
        pred (torch.Tensor): Predicted binary mask.
        target (torch.Tensor): Target binary mask.

    Returns:
        torch.Tensor: Dice coefficient accuracy score.
    """
    # Create mask for pred and targets
    pred_mask = pred > 0.5
    target_mask = target > 0.5

    # Calculate intersection and union base on the mask AND & OR operation
    intersection = torch.sum(pred_mask * target_mask)
    total = torch.sum(pred_mask) + torch.sum(target_mask)

    # Dice coefficient formula
    dice = 2 * intersection / total
    return dice