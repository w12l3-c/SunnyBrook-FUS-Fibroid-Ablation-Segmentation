import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from PIL import Image

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

# --------------------- Pytorch UNet++ --------------------- #
def auto_UNETPP(in_channels, num_classes):
    model = smp.UnetPlusPlus(
        encoder_name="resnet101",       
        encoder_weights="imagenet",    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )
        
    return model

def prepare_transform():
    params = get_preprocessing_params('resnet101', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.3),
        torchvision.transforms.ToTensor(),
    ])
    
    return transform

def prepare_loss(option='BCE'):
    if option == 'BCE':
        criterion = nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([1.0, 5.0])
    if option == 'CE':
        criterion = nn.CrossEntropyLoss()
    if option == 'Dice_Binary':
        criterion = smp.losses.DiceLoss('binary')
    if option == 'Dice_Multi':
        criterion = smp.losses.DiceLoss('multilabel')
        
    return criterion

def prepare_optimizer(model, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def prepare_scheduler(optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=verbose)
    return scheduler

# --------------------- Accuracy --------------------- #
def accuracy_iou(pred, target):
    pred_mask = pred == 1
    target_mask = target == 1

    intersection = torch.logical_and(pred_mask, target_mask).sum()
    union = torch.logical_or(pred_mask, target_mask).sum()

    iou = intersection / union
    return iou

def accuracy_intersect(pred, target):
    pred_mask = pred == 1
    target_mask = target == 1

    # Calculate intersection only for class 1
    intersection = torch.logical_and(pred_mask, target_mask).sum()
    return intersection / pred.sum()


# --------------------- Class Weights ------------------------ #
def calculate_weights(mask):
  total = mask.numel()
  pos = torch.sum(mask > 0.5)
  return total/pos


# --------------------- Inference --------------------- #
def predict(model, dataset, device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
    
    model.eval()
    with torch.no_grad():
        for pair in dataset:
            image = pair['img']
            mask = pair['mask']
            
            image = image.convert('RGB')
            resized_image = image.resize((320, 320))
            transformed_image = transform(resized_image).to(device)
            logits = model(transformed_image.unsqueeze(0))
            pred = torch.softmax(logits, dim=1).argmax(dim=1).float()
            pred = pred.squeeze().cpu().numpy() * 255
            pred = cv2.resize(pred, (image.size[0], image.size[1]))
            
            yield (image, pred)
            
def predict_UNET(model, dataset, device):
    generator = predict(model, dataset, device)
    for prediction in generator:
        image, pred = prediction
        plt.figure(figsize=(10,10))
        plt.title('Prediction')
        plt.imshow(image, alpha=0.8)
        plt.imshow(pred, alpha=0.2, cmap='gray')
        plt.show()