import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torchvision

from sklearn.model_selection import train_test_split

import os 
import numpy as np
import pydicom
import random
import time
import datetime
import matplotlib.pyplot as plt

from utils import *
from dataloader import *
from save_load import *
from train import deeplabv3_train_model, unet_train_model   

from models import DeepLabV3, Unet, Unetpp, DeepLabV3plus, FPN, MAnet


def deeplabv3_inference(test_dataset, model_path, device):
    # Prepare dataset
    dataset = test_dataset
    
    # Load model
    model, transform, loss_fn, optimizer, scheduler = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
    model = load_model_torch(model, model_path)
    
    # Inference
    model.eval()
    img_path = dataset[0]["img"]
    mask_path = dataset[0]["mask"]
    
    image = Image.open(img_path)
    img = image.convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img)
    img_batch = img_tensor.unsqueeze(0)
    img_batch = img_batch.to(device)
    
    with torch.no_grad():
        output_logits = model(img_batch)['out'][0]
    output_pred = torch.sigmoid(output_logits) * 255
    
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.randn(1, ) * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(output_pred.byte().cpu().numpy()).resize(img.size)
    r = r.convert("RGB")
    r.putpalette(colors)

    plt.imshow(img, alpha=0.9)
    plt.imshow(r, alpha=0.5)
    
    
def unet_inference(test_dataset, model_path, device, img_size=(320, 320), display=True):
    # Set Seed
    torch.manual_seed(42)
    
    model = Unet.auto_UNET(in_channels=3, num_classes=2)
    model = load_model_torch(model, model_path)
    model = model.to(device)
    
    # Run Inference function
    Unet.predict_UNET(model, test_dataset, device, img_size, display)
    
    
def unetpp_inference(test_dataset, model_path, device, img_size=(320, 320)):
    # Set Seed
    torch.manual_seed(42)
    
    model = Unetpp.auto_UNETPP(in_channels=3, num_classes=2)
    model = load_model_torch(model, model_path)
    model = model.to(device)
    
    # Run Inference function
    Unetpp.predict_UNETPP(model, test_dataset, device, img_size)
    
    
def deeplabv3plus_inference(test_dataset, model_path, device, img_size=(320, 320)):
    # Set Seed
    torch.manual_seed(42)
    
    model = DeepLabV3plus.auto_DEEPLABV3P(in_channels=3, num_classes=2)
    model = load_model_torch(model, model_path)
    model = model.to(device)
    
    # Run Inference function
    DeepLabV3plus.predict_DEEPLABV3P(model, test_dataset, device, img_size)
    
def fpn_inference(test_dataset, model_path, device, img_size=(320, 320)):
    # Set Seed
    torch.manual_seed(42)
    
    model = FPN.auto_FPN(in_channels=3, num_classes=2)
    model = load_model_torch(model, model_path)
    model = model.to(device)
    
    # Run Inference function
    FPN.predict_FPN(model, test_dataset, device, img_size)