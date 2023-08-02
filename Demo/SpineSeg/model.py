import torch
import torchvision

from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

def create_Unet(in_channels=3, num_classes=1):
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