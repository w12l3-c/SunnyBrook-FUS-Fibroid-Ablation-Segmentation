# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the DeepLabV3+ model
# =============================================================================

# =================== Imports =================== #
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import uuid
import time
import pydicom
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

import torchvision

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

# =================== DeepLabV3+ =================== #
def auto_DEEPLABV3P(in_channels, num_classes, encoder_name='resnet50', encoder_weights='imagenet'):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,       
        encoder_weights=encoder_weights,    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )
    
    return model

def prepare_transform(flip=0.3):
    params = get_preprocessing_params('resnet50', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(flip),
        torchvision.transforms.ToTensor(),
    ])
    
    return transform

def prepare_loss(option='BCE'):
    if option == 'BCE':
        criterion = nn.BCEWithLogitsLoss()  
    if option == 'CE':
        criterion = nn.CrossEntropyLoss()
    if option == 'Dice_Binary':
        criterion = smp.losses.DiceLoss('binary')
    if option == 'Dice_Multi':
        criterion = smp.losses.DiceLoss('multilabel')
        
    return criterion

def prepare_optimizer(model, lr=1e-3, option='Adam'):
    if option == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if option == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if option == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return optimizer

def prepare_scheduler(optimizer, factor=0.1, patience=10, min_lr=1e-6, verbose=True):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=verbose)
    return scheduler


# ===================== Accuracy ===================== #
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

def accuracy_dice(pred, target):
    pred_mask = pred > 0.5
    target_mask = target > 0.5

    intersection = torch.sum(pred_mask * target_mask)
    total = torch.sum(pred_mask) + torch.sum(target_mask)

    dice = 2 * intersection / total
    return dice

def accuracy_iou_multi(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for i in torch.unique(target):
        pred_mask = pred == i
        target_mask = target == i

        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask + target_mask)

        iou = intersection / union
        ious.append(iou)
    
    return torch.mean(torch.tensor(ious))


# ===================== Class Weights ===================== #
def calculate_weights(mask):
  total = mask.numel()
  pos = torch.sum(mask > 0.5)
  return total/pos


# ====================== Gamma Correction ====================== #
def gamma_correction_cv2(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image, table.astype(np.uint8))

def gamma_correction_pil(image, gamma=1.0):
    enhancer = ImageEnhance.Brightness(image)
    gamma_corrected_image = enhancer.enhance(gamma)
    return gamma_corrected_image


# ====================== Inference ====================== #
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
            resized_image = gamma_correction_pil(resized_image, gamma=1.5)
            #resized_image = ImageOps.equalize(resized_image)
            transformed_image = transform(resized_image).to(device)
            
            start_time = time.time()
            logits = model(transformed_image.unsqueeze(0))
            pred = torch.softmax(logits, dim=1).argmax(dim=1).float()
            end_time = time.time()
            
            totensor = torchvision.transforms.ToTensor()
            acc_iou = accuracy_iou(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            acc_basic = accuracy_basic(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            acc_dice = accuracy_dice(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            
            pred = pred.squeeze().cpu().numpy() * 255
            pred = cv2.resize(pred, img_size)
            
            mask = mask.resize(img_size)
            
            inference_time = end_time - start_time
            
            yield (image, mask, pred, (acc_iou, acc_basic, acc_dice), inference_time)
            
def predict_UNET(model, dataset, device, img_size=(320, 320), display=True):
    generator = predict(model, dataset, device, img_size)
    if display:
        save = input('Save predictions? (y/n): ')
        directory = '/mnt/HDD_1TB/Wallace/Code/Seg2D/predictions/'
    
        if save == 'y':
            if os.path.exists(directory):
                print('Directory exists -- Continue')
            else:
                os.makedirs(directory)
                print('Directory created')

        for i, prediction in enumerate(generator):
            image, mask, pred, acc, time = prediction
            acc_iou, acc_basic, acc_dice = acc
            image = gamma_correction_pil(image, gamma=1.5)  
            
            mask_edge = cv2.Canny(np.asarray(mask), 100, 200)
            pred_edge = cv2.Canny(pred.astype(np.uint8), 100, 200)
            
            mask_edge_red = np.zeros((mask_edge.shape[0], mask_edge.shape[1], 3))
            mask_edge_red[mask_edge > 0] = [255, 0, 0]
            mask_edge_red = mask_edge_red.astype(np.uint8)
            pred_edge_green = np.zeros((pred_edge.shape[0], pred_edge.shape[1], 3))
            pred_edge_green[pred_edge > 0] = [0, 255, 0]
            pred_edge_green = pred_edge_green.astype(np.uint8)
            
            
            edge_overlay = mask_edge_red + pred_edge_green
            edge_overlay = edge_overlay.astype(np.uint8)
            
            mask_red = np.zeros((mask.size[1], mask.size[0], 3))
            pred_green = np.zeros((mask.size[1], mask.size[0], 3))
            
            mask_map = np.asarray(mask) > 128
            pred_map = pred > 0.5
            
            mask_red[mask_map] = [255, 0, 0]
            pred_green[pred_map] = [0, 255, 0]
            
            mask_red = mask_red.astype(np.uint8)
            pred_green = pred_green.astype(np.uint8)
            
            mask_overlay = mask_red + pred_green
            mask_overlay = mask_overlay.astype(np.uint8)
            
            fig, ax = plt.subplots(2,3, figsize=(20,15))
            
            ax[0][0].imshow(image)
            ax[0][1].imshow(image, alpha=0.7)
            ax[0][1].imshow(mask_edge_red, alpha=0.3)
            ax[0][2].imshow(image, alpha=0.7)
            ax[0][2].imshow(pred_edge_green, alpha=0.3)
            
            ax[1][0].imshow(mask_red)
            ax[1][1].imshow(pred_green)
            ax[1][2].imshow(edge_overlay)
    
            ax[0][0].set_title('Image')
            ax[0][1].set_title('Mask Overlay')
            ax[0][2].set_title('Pred Overlay')
            
            ax[1][0].set_title('Ground Truth')
            ax[1][1].set_title(f'Prediction: {acc_iou*100:.2f}(IOU) | {acc_basic*100:.2f}(Acc)')
            ax[1][2].set_title('Edge Overlay')
            
            ax[0][0].axis('off')
            ax[0][1].axis('off')
            ax[0][2].axis('off')
            ax[1][0].axis('off')
            ax[1][1].axis('off')
            ax[1][2].axis('off')
            
            fig.suptitle(f'Inference Time: {time:.4f} seconds')
            plt.show()
            
            if save == 'y':
                filename = f"predictions/mask_{i}.jpg"
                # cv2.imwrite(os.path.join(directory, filename), pred)
                fig.savefig(filename)
                
            if i % 10 == 0:
                quit = input('Exit? (y/n): ')
                if quit == 'y':
                    break
            
        print('Inference Complete')
        
    else:
        acc_ious = []
        acc_basics = []
        acc_dices = []
        times = []
        for prediction in generator:
            image, mask, pred, acc, time = prediction
            acc_iou, acc_basic, acc_dice = acc
            acc_ious.append(acc_iou.item())
            acc_basics.append(acc_basic.item())
            acc_dices.append(acc_dice.item())
            times.append(time)
            
        fig, ax = plt.subplots(2, 2, figsize=(20,10))
        
        ax[0][0].set_title(f'Accuracy (IOU) | Median:{np.mean(np.array(acc_ious))*100:.4f}')
        ax[0][0].plot(acc_ious)
        ax[0][1].set_title(f'Accuracy (Basic) | Median:{np.mean(np.array(acc_basics))*100:.4f}')
        ax[0][1].plot(acc_basics)
        ax[1][0].set_title(f'Accuracy (Dice) | Median:{np.mean(np.array(acc_dices))*100:.4f}')
        ax[1][0].plot(acc_dices)
        ax[1][1].set_title(f'Inference Time (s) | Median:{np.mean(np.array(times)):.4f}')
        ax[1][1].plot(times)
        
        plt.show()
            
        

