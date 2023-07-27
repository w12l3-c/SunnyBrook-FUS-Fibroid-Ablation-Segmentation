import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

# --------------------- Tensorflow UNet --------------------- #
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
    

# --------------------- Pytorch UNet --------------------- #
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
    
    
# --------------------- Pytorch UNet --------------------- #
def pretrained_UNET(in_channels, num_classes):
    model = smp.Unet(
        encoder_name="resnet34",       
        encoder_weights="imagenet",    
        in_channels=in_channels,                  
        classes=num_classes,                      
    )
    return model

def prepare_UNET(model):
    params = get_preprocessing_params('resnet34', pretrained='imagenet')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**params),
    ])
    
    loss_fn = smp.losses.DiceLoss('bilinear')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    
    return transform
   
    