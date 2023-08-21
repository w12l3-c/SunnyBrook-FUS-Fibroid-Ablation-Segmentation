# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the Unet model
# Options for the Unet model are:
#   - Default Tensorflow Unet
#   - Default Pytorch Unet
#   - Modifiable Pytorch Unet
# =============================================================================

# =================== Imports =================== #
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import uuid
import time
import pydicom
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

import torchvision

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

# ====================== Tensorflow UNet Classsic ====================== #
def conv_block(input, in_Channels):
    conv1 = layers.Conv2D(in_Channels, (3,3), activation='relu', padding='same')(input)
    batch1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv2D(in_Channels, (3,3), activation='relu', padding='same')(batch1)
    return conv2
    
def down_block(input, in_Channels):
    conv = conv_block(input, in_Channels)
    pool = layers.MaxPooling2D((2, 2))(conv)
    return conv, pool
        
def up_block(input, in_Channels, skip):
    up = layers.Conv2DTranspose(in_Channels, (2, 2), strides=(2, 2), padding='same')(input)
    cat = layers.concat([up, skip], axis=-1)
    conv = conv_block(cat, in_Channels)
    return conv

def basic_UNET(input_shape):
    input_layer = layers.Input(shape=input_shape)
    
    skip1, down1 = down_block(input_layer, 64)
    skip2, down2 = down_block(down1, 128)
    skip3, down3 = down_block(down2, 256)
    skip4, down4 = down_block(down3, 512)
    
    bottle = conv_block(down4, 1024)
    
    up1 = up_block(bottle, 512, skip4)
    up2 = up_block(up1, 256, skip3)
    up3 = up_block(up2, 128, skip2)
    up4 = up_block(up3, 64, skip1)
    
    output = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up4)

    model = keras.models.Model(inputs=input_layer, outputs=output)
    
    return model
    

# ====================== Pytorch UNet Classic ====================== #
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn1(self.conv2(x)))
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        out = self.down(x)
        out = self.conv(out)
        return out, x
               
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        cat = torch.cat([x, skip], dim=1)
        x = self.conv(cat)
        return x
        
class UNet(nn.Module):    
    def __init__(self, in_channels=1, num_classes=1, channels = [64, 128, 256, 512, 1024]):
        super().__init__()
        self.ups = nn.ModuleList()
        for i in range(4, 0, -1):
            self.ups.append(UpBlock(channels[i], channels[i-1]))
        
        self.downs = nn.ModuleList()
        for i in range(4):
            self.downs.append(DownBlock(channels[i], channels)[i])
            
        self.input = ConvBlock(in_channels, channels[0])
        self.output = nn.Conv2d(channels[0], 2, kernel_size=(1,1))
        
        if num_classes > 1:
            self.pred = nn.Softmax(dim=1)
        else:
            self.pred = nn.Sigmoid()
        
    def forward(self, x):
        skips = []
        
        x = self.input(x)
        
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)
            
        for i, up in enumerate(self.ups):
            skip = skips[i]
            x = up(x, skip)
            
        x = self.output(x)
        x = self.pred(x)
        
        return x
    
    
# ====================== Pytorch UNet Modifiable ====================== #
from torchinfo import summary
def auto_UNET(in_channels, num_classes):
    """
    Create a UNet model with the specified number of input channels and output classes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: UNet model.
    """
    # I am using smp here because they have pretrained weights from imagenet
    model = smp.Unet(
        encoder_name="resnet101",       
        encoder_weights="imagenet",    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )

    # Freezing weights is not recommended, the pretrained weights are just to make the convergence happen faster
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # for param in model.decoder.parameters():
    #     param.requires_grad = False
        
    return model

def prepare_transform(flip=0.3):
    """
    Prepare data augmentation transformations.

    Args:
        flip (float, optional): Probability of horizontal flip. Default is 0.3.

    Returns:
        torchvision.transforms.Compose: Data transformation pipeline.
    """
    params = get_preprocessing_params('resnet101', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(flip),
        torchvision.transforms.ToTensor(),
    ])
    
    return transform

def prepare_loss(option='BCE'):
    """
    Prepare the loss function.

    Args:
        option (str, optional): Loss function option ('BCE', 'CE', 'Dice_Binary', 'Dice_Multi'). Default is 'BCE'.

    Returns:
        torch.nn.Module: Loss function.
    """
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
    """
    Prepare the optimizer.

    Args:
        model (torch.nn.Module): Model for optimization.
        lr (float, optional): Learning rate. Default is 1e-3.
        option (str, optional): Optimizer option ('Adam', 'AdamW', 'SGD'). Default is 'Adam'.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    if option == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if option == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if option == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return optimizer

def prepare_scheduler(optimizer, factor=0.1, patience=10, min_lr=1e-6, verbose=True):
    """
    Prepare the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for which to schedule learning rates.
        factor (float, optional): Factor by which to reduce learning rate. Default is 0.1.
        patience (int, optional): Number of epochs with no improvement before reducing learning rate. Default is 10.
        min_lr (float, optional): Minimum learning rate. Default is 1e-6.
        verbose (bool, optional): If True, print a message when learning rate is reduced. Default is True.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: Learning rate scheduler.
    """
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
            
def predict_UNET(model, dataset, device, img_size=(320, 320), display=True):
    """
    Perform predictions using a UNet model on a dataset and optionally display the results.
    If `display` is `True`, the user will be prompted to save the predictions of individual inference.
    If `display` is `False`, it will save a figure where it find the overall metrics for the entire dataset.

    Args:
        model (torch.nn.Module): UNet model for segmentation.
        dataset (iterable): Iterable containing image-mask pairs.
        device (torch.device): Device to run inference on.
        img_size (tuple): Size to resize input images.
        display (bool): Whether to display results interactively (default is True).
    """
    # Create generator for predictions
    generator = predict(model, dataset, device, img_size)
    
    # If display is True, prompt user to save predictions
    if display:
        # Prompt user to save predictions
        save = input('Save predictions? (y/n): ')
        directory = '/mnt/HDD_1TB/Wallace/Code/Seg2D/predictions/'

        # Create directory if it doesn't exist
        if save == 'y':
            if os.path.exists(directory):
                print('Directory exists -- Continue')
            else:
                os.makedirs(directory)
                print('Directory created')

        # Iterate over generator
        for i, prediction in enumerate(generator):
            # Unpack prediction
            image, mask, pred, acc, time = prediction
            acc_iou, acc_basic, acc_dice = acc
            # image = gamma_correction_pil(image, gamma=1.5)  
            
            # If the mask is avaliable do mask operations
            if mask is not None:
                # Grab the contour of the mask and prediction
                mask_edge = cv2.Canny(np.asarray(mask), 100, 200)
                pred_edge = cv2.Canny(pred.astype(np.uint8), 100, 200)
                
                # Recolor them into red and green
                mask_edge_red = np.zeros((mask_edge.shape[0], mask_edge.shape[1], 3))
                mask_edge_red[mask_edge > 0] = [255, 0, 0]
                mask_edge_red = mask_edge_red.astype(np.uint8)
                pred_edge_green = np.zeros((pred_edge.shape[0], pred_edge.shape[1], 3))
                pred_edge_green[pred_edge > 0] = [0, 255, 0]
                pred_edge_green = pred_edge_green.astype(np.uint8)
                
                # Overlay them such that the overlapping becomes yellow
                edge_overlay = mask_edge_red + pred_edge_green
                edge_overlay = edge_overlay.astype(np.uint8)
                
                # Recolouring the prediction and mask in green and red respectively
                mask_red = np.zeros((mask.size[1], mask.size[0], 3))
                pred_green = np.zeros((mask.size[1], mask.size[0], 3))
                
                mask_map = np.asarray(mask) > 128
                pred_map = pred > 0.5
                
                mask_red[mask_map] = [255, 0, 0]
                pred_green[pred_map] = [0, 255, 0]
                
                mask_red = mask_red.astype(np.uint8)
                pred_green = pred_green.astype(np.uint8)
                
                # Overlay them so make the overlapping yellow
                mask_overlay = mask_red + pred_green
                mask_overlay = mask_overlay.astype(np.uint8)
            
            # Grid plot of image, mask, and prediction
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
            
            # Save the prediction if user wants to
            if save == 'y':
                filename = f"predictions/mask_{i}.jpg"
                # cv2.imwrite(os.path.join(directory, filename), pred)
                fig.savefig(filename)
            
            # Prompt user to exit every 10 images
            if i % 10 == 0:
                quit = input('Exit? (y/n): ')
                if quit == 'y':
                    break
            
        print('Inference Complete')
    
    # If display is False, return accuracy metrics
    else:
        # Save accuracy metrics
        acc_ious = []
        acc_basics = []
        acc_dices = []
        times = []
        
        # Iterate over generator
        for prediction in generator:
            image, mask, pred, acc, time = prediction
            acc_iou, acc_basic, acc_dice = acc
            acc_ious.append(acc_iou.item())
            acc_basics.append(acc_basic.item())
            acc_dices.append(acc_dice.item())
            times.append(time)
        
        # Plot accuracy metrics
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
        
        # Save the metrics
        save_path = f'predictions/metrics_{uuid.uuid1}.png'
        fig.savefig(save_path)
        print(f'Metrics saved to {save_path}')
        