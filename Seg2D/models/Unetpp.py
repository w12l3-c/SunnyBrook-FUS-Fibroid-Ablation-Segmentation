# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the Unet++ model
# =============================================================================

# =================== Imports =================== #
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import os
import time
import cv2
import uuid
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from PIL import Image, ImageOps, ImageEnhance

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

# ===================== Pytorch UNet++ ===================== #
def auto_UNETPP(in_channels, num_classes):
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",       
        encoder_weights="imagenet",    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )
        
    return model

def prepare_transform():
    params = get_preprocessing_params('resnet50', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.3),
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

def prepare_optimizer(model, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def prepare_scheduler(optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=verbose)
    return scheduler

# ====================== Accuracy ====================== #
def accuracy_iou(pred, target):
    """
    Calculate Intersection over Union (IoU) accuracy between predicted and target binary masks.

    Args:
        pred (torch.Tensor): Predicted binary mask.
        target (torch.Tensor): Target binary mask.

    Returns:
        torch.Tensor: IoU accuracy score.
    """
    # Create mask for pred and targets
    pred_mask = pred > 0.5  
    target_mask = target > 0.5

    # Calculate intersection and union base on the mask AND & OR operation
    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask + target_mask)

    # Calculate IoU
    iou = intersection / union
    return iou

def accuracy_intersect(pred, target):
    """
    Calculate accuracy based on the intersection of predicted and target binary masks.

    Args:
        pred (torch.Tensor): Predicted binary mask.
        target (torch.Tensor): Target binary mask.

    Returns:
        torch.Tensor: Intersection accuracy score.
    """
    # Create mask for pred and targets  
    pred_mask = pred > 0.5
    target_mask = target > 0.5
    
    # Using the AND operation to calculate the intersection
    intersection = torch.sum(pred_mask * target_mask)
    pred_total = torch.sum(pred_mask == True)
    
    return intersection / pred_total

def accuracy_basic(pred, target):
    """
    Calculate basic accuracy between predicted and target binary masks.

    Args:
        pred (torch.Tensor): Predicted binary mask.
        target (torch.Tensor): Target binary mask.

    Returns:
        torch.Tensor: Basic accuracy score.
    """
    # Create mask for pred and targets
    pred = pred > 0.5
    target = target > 0.5
    # Simply sum up the number of correct pixels and divide by the total number of pixels
    correct = torch.sum(pred == target)
    return correct / pred.numel()

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

def accuracy_iou_multi(pred, target):
    """
    Calculate Intersection over Union (IoU) accuracy for multiple classes in predicted and target masks.

    Args:
        pred (torch.Tensor): Predicted multi-class mask.
        target (torch.Tensor): Target multi-class mask.

    Returns:
        torch.Tensor: Mean IoU accuracy score across classes.
    """
    pred = torch.argmax(pred, dim=1)    
    ious = []
    # Loop through all classes
    for i in torch.unique(target):
        # Same operation as th regular IOU
        pred_mask = pred == i
        target_mask = target == i

        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask + target_mask)

        iou = intersection / union
        ious.append(iou)
    
    # Return the mean of all classes' IOU
    return torch.mean(torch.tensor(ious))


# ====================== Class Weights ======================--- #
def calculate_weights(mask):
    """
    Calculate class weights based on the provided mask.

    Args:
        mask (torch.Tensor): Binary mask.

    Returns:
        torch.Tensor: Class weights.
    """ 
    total = mask.numel()
    pos = torch.sum(mask > 0.5)
    return total/pos

# ====================== Gamma Correction ====================== #
def gamma_correction_cv2(image, gamma=1.0):
    """
    Apply gamma correction to an image using OpenCV.

    Args:
        image (numpy.ndarray): Input image.
        gamma (float): Gamma correction factor (default is 1.0).

    Returns:
        numpy.ndarray: Gamma-corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image, table.astype(np.uint8))

def gamma_correction_pil(image, gamma=1.0):
    """
    Apply gamma correction to an image using PIL (Python Imaging Library).

    Args:
        image (PIL.Image.Image): Input image.
        gamma (float): Gamma correction factor (default is 1.0).

    Returns:
        PIL.Image.Image: Gamma-corrected image.
    """
    enhancer = ImageEnhance.Brightness(image)
    gamma_corrected_image = enhancer.enhance(gamma)
    return gamma_corrected_image

# ====================== Inference ====================== #
def predict(model, dataset, device, img_size):
    """
    Generate predictions using a PyTorch model on a dataset.

    Args:
        model (torch.nn.Module): PyTorch model for segmentation.
        dataset (iterable): Iterable containing image-mask pairs.
        device (torch.device): Device to run inference on.
        img_size (tuple): Size to resize input images.

    Yields:
        tuple: Tuple containing (image, ground truth mask, predicted mask, accuracy scores, inference time).
    """
    # Transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
    
    # Set model to evaluation mode
    model.eval()
    # Inference mode
    with torch.no_grad():
        for pair in dataset:
            # Get image and mask
            image = pair['img']
            mask = pair['mask']
            
            # Preprocessing
            image = image.convert('RGB')
            resized_image = image.resize(img_size)
            resized_image = gamma_correction_pil(resized_image, gamma=1.5)
            #resized_image = ImageOps.equalize(resized_image)
            transformed_image = transform(resized_image).to(device)
            
            # Inference
            start_time = time.time()
            logits = model(transformed_image.unsqueeze(0))
            pred = torch.softmax(logits, dim=1).argmax(dim=1).float()
            end_time = time.time()
            
            # Calculate accuracy and metrics
            totensor = torchvision.transforms.ToTensor()
            acc_iou = accuracy_iou(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            acc_basic = accuracy_basic(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            acc_dice = accuracy_dice(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            
            pred = pred.squeeze().cpu().numpy() * 255
            pred = cv2.resize(pred, img_size)
            
            # Resize mask
            if mask is not None:
                mask = mask.resize(img_size)
            
            # Calculate inference time
            inference_time = end_time - start_time
            
            yield (image, mask, pred, (acc_iou, acc_basic, acc_dice), inference_time)
            
def predict_UNETPP(model, dataset, device, img_size=(320, 320), display=True):
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
            # image = gamma_correction_pil(image, gamma=1.5)  
            
            if mask is not None:
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
        
        ax[0][0].set_title(f'Accuracy (IOU) | Median:{np.median(np.array(acc_ious))*100:.4f}')
        ax[0][0].plot(acc_ious)
        ax[0][1].set_title(f'Accuracy (Basic) | Median:{np.median(np.array(acc_basics))*100:.4f}')
        ax[0][1].plot(acc_basics)
        ax[1][0].set_title(f'Accuracy (Dice) | Median:{np.median(np.array(acc_dices))*100:.4f}')
        ax[1][0].plot(acc_dices)
        ax[1][1].set_title(f'Inference Time (s) | Median:{np.median(np.array(times)):.4f}')
        ax[1][1].plot(times[1:])
        
        plt.show()
        
        save_path = f'predictions/metrics_{uuid.uuid1}.png'
        fig.savefig(save_path)
        print(f'Metrics saved to {save_path}')