import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import uuid
import time
import pydicom
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torchvision

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params


def auto_DEEPLABV3P(in_channels, num_classes, encoder_name='resnet101', encoder_weights='imagenet'):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,       
        encoder_weights=encoder_weights,    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )
    
    return model

def prepare_transform():
    params = get_preprocessing_params('resnet101', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.2),
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
    pred_mask = pred > 0.5
    target_mask = target > 0.5

    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask + target_mask)

    iou = intersection / union
    return iou

def accuracy_intersect(pred, target):
    pred_mask = pred > 0.5
    target_mask = target > 0.5
    intersection = torch.sum(pred_mask * target_mask)
    pred_total = torch.sum(pred_mask == True)
    
    return intersection / pred_total

def accuracy_basic(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    correct = torch.sum(pred == target)
    return correct / pred.numel()


# --------------------- Class Weights ------------------------ #
def calculate_weights(mask):
  total = mask.numel()
  pos = torch.sum(mask > 0.5)
  return total/pos


# --------------------- Inference --------------------- #
def predict(model, dataset, device, img_size):
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
            resized_image = image.resize(img_size)
            transformed_image = transform(resized_image).to(device)
            
            start_time = time.time()
            logits = model(transformed_image.unsqueeze(0))
            pred = torch.softmax(logits, dim=1).argmax(dim=1).float()
            end_time = time.time()
            
            totensor = torchvision.transforms.ToTensor()
            acc = accuracy_iou(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            
            pred = pred.squeeze().cpu().numpy() * 255
            pred = cv2.resize(pred, img_size)
            
            mask = mask.resize(img_size)
            
            inference_time = end_time - start_time
            
            yield (image, mask, pred, acc, inference_time)
            
            
def predict_DEEPLABV3P(model, dataset, device, img_size=(320, 320)):
    generator = predict(model, dataset, device, img_size)
    save = input('Save predictions? (y/n): ')
    directory = '/mnt/HDD_1TB/Wallace/Code/Seg2D/predictions/'
    
    if save == 'y':
        try:
            if os.path.exists(directory):
                print('Directory exists -- Continue')
        except Exception as e:
            os.mkdir(directory)
            print('Directory created')
    
    for i, prediction in enumerate(generator):
        image, mask, pred, acc, time = prediction
        
        fig, ax = plt.subplots(1,4, figsize=(20,15))
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap='gray')
        ax[2].imshow(pred, cmap='gray')
        ax[3].imshow(image, alpha=0.7)
        ax[3].imshow(pred, alpha=0.3, cmap='gray')
        
        ax[0].set_title('Image')
        ax[1].set_title('Ground Truth')
        ax[2].set_title(f'Prediction: {acc:.2f}')
        ax[3].set_title('Overlay')
        
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        
        fig.suptitle(f'Inference Time: {time:.4f} seconds')
        plt.show()
        
        if save == 'y':
            filename = f"mask_{i}.jpg"
            cv2.imwrite(os.path.join(directory, filename), pred)
        
        if i % 10 == 0:
            quit = input('Exit? (y/n): ')
            if quit == 'y':
                break
        
    print('Inference Complete')
