import os

# Set TF_CPP_MIN_LOG_LEVEL to 2 to suppress warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import uuid
import time
import pydicom
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

import torchvision

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

# --------------------- Tensorflow UNet Classsic --------------------- #
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
    

# --------------------- Pytorch UNet Classic --------------------- #
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
    
    
# --------------------- Pytorch UNet Library --------------------- #
def auto_UNET(in_channels, num_classes):
    model = smp.Unet(
        encoder_name="resnet101",       
        encoder_weights="imagenet",    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )
    
    # Somehow freezing weights decrease performance
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # for param in model.decoder.parameters():
    #     param.requires_grad = False
        
    return model

def prepare_transform(flip=0.3):
    params = get_preprocessing_params('resnet101', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(flip),
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

def prepare_scheduler(optimizer, factor=0.1, patience=10, min_lr=1e-6, verbose=True):
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

# --------------------- Gamma Correction --------------------- #
def gamma_correction_cv2(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image, table.astype(np.uint8))

def gamma_correction_pil(image, gamma=1.0):
    enhancer = ImageEnhance.Brightness(image)
    gamma_corrected_image = enhancer.enhance(gamma)
    return gamma_corrected_image

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
            acc_iou = accuracy_iou(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            acc_basic = accuracy_basic(pred, totensor(mask.convert('L').resize(img_size)).to(device))
            
            pred = pred.squeeze().cpu().numpy() * 255
            pred = cv2.resize(pred, img_size)
            
            mask = mask.resize(img_size)
            
            inference_time = end_time - start_time
            
            yield (image, mask, pred, acc_iou, acc_basic, inference_time)
            
def predict_UNET(model, dataset, device, img_size=(320, 320)):
    generator = predict(model, dataset, device, img_size)
    save = input('Save predictions? (y/n): ')
    directory = '/mnt/HDD_1TB/Wallace/Code/Seg2D/predictions/'
 
    if save == 'y':
        if os.path.exists(directory):
            print('Directory exists -- Continue')
        else:
            os.makedirs(directory)
            print('Directory created')

    
    for i, prediction in enumerate(generator):
        image, mask, pred, acc_iou, acc_basic, time = prediction
        image = gamma_correction_pil(image, gamma=1.5)  
        
        fig, ax = plt.subplots(1,4, figsize=(20,15))
        
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap='gray')
        ax[2].imshow(pred, cmap='gray')
        ax[3].imshow(image, alpha=0.7)
        ax[3].imshow(pred, alpha=0.3, cmap='gray')
        
        ax[0].set_title('Image')
        ax[1].set_title('Ground Truth')
        ax[2].set_title(f'Prediction: {acc_iou*100:.2f}(IOU) | {acc_basic*100:.2f}(Acc)')
        ax[3].set_title('Overlay')
        
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        
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
        
