# ======================================================================== #
# File Description:
# ------------------
# This file is used for train or inferencing model on the Sagittal Dataset
# It is a multilabel segmentation task for every region in sagittal
#
# This is more of a future work:
# This is not going to perform well on the current dataset as not every picture
# has all the regions segmented
#
# Comment out the training or inference code to run the other
# ======================================================================== #

# =================== Imports =================== #
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
from models import DeepLabV3, Unet, Unetpp, DeepLabV3plus, FPN, MAnet
from model_run_fn import deeplabv3_run, unet_run, unetpp_run, deeplabv3p_run, fpn_run, manet_run
from model_inference_fn import deeplabv3_inference, unet_inference, unetpp_inference, deeplabv3plus_inference

# =================== Hyperparameters =================== #
NUM_WORKERS = os.cpu_count() # Number of CPU cores used for data loading
PIN_MEMORY = True   # Pin memory for faster GPU transfer
NUM_EPOCHS = 200 # Just fot test, in pratical should be 100 or more
BATCH_SIZE = 4  # Between 8-16 is good
IN_CHANNELS = 3 # RGB
COLOR_DICT = {'Background':(256, 256, 256), 'Spine': (35, 132, 250), 'Bowel': (14, 240, 56), 'Muscle': (214, 51, 36), 'Skin': (240, 170, 31), 'hip_L': (173, 20, 250), 'hip_R': (131, 20, 250)}   # Color map in dictionart
COLOR_LIST = [v for v in COLOR_DICT.values()]   # Colour map in list
ID2LABEL = {i: k for i, (k, v) in enumerate(COLOR_DICT.items())}
LABEL2ID = {k: i for i, (k, v) in enumerate(COLOR_DICT.items())}
NUM_CLASSES = len(COLOR_LIST) # Classes to Segment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'nvidia/segformer-b0-finetuned-ade-512-512'  # Huggingface model path

# =================== Patient Dataset =================== #
# Set seed for reproducibility
torch.manual_seed(42)

# Define which directories are part of the dataset
train_path_list = listsiemens[:9] + listbones[1:] + listlowres + listarrayus[1:]
test_path_list = [listsiemens[-1]] + [listarrayus[0]] + [listbones[0]]

# Create a list of patient class objects
train_patients = create_patient_list(path_list=train_path_list)
test_patients = create_patient_list(path_list=test_path_list)

# =================== Sagittal Datasets =================== #
# Grab Each Patient's Sagittal image and mask list
train_sag_dataset = get_multilabel_sagittal(train_patients)
test_sag_dataset = get_multilabel_sagittal(test_patients)
train_sag_dataset, val_sag_dataset = train_test_split(train_sag_dataset, test_size=0.1, random_state=42)

# Convert the pydicom and mask list to PIL images
train_sag_dataset = convert_to_PIL_multi(train_sag_dataset, COLOR_DICT)
val_sag_dataset = convert_to_PIL_multi(val_sag_dataset, COLOR_DICT)
test_sag_dataset = convert_to_PIL_multi(test_sag_dataset, COLOR_DICT)

if __name__ == '__main__':
    ...
    # Currently the training loop and functions are all catered for background and foreground so either 
    # make new functions or change the current ones to accomodate for multilabel segmentation
    
    # =================== Training =================== #
    # try:
    #     # Set the writer and save path
    #     unet_writer = f"runs/Unet_Sagittal_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    #     unet_save_path = f"./trained_models/Unet_Sagittal_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    #     unet_run(train_sag_dataset, val_sag_dataset, device, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, True, unet_writer, unet_save_path, 'BCE')
    # except Exception as e:
    #     print(e)
    #     print('This training sessions failed')
    
    # =================== Inference =================== #
    # try:
    #     # Set the path where you save your model
    #     unet_save_path = './trained_models/Unet_Sagittal_2023-08-01_18:02:56.pth'
    #     unet_inference(test_sag_dataset, unet_save_path, device, num_classes=NUM_CLASSES, display=True)
    # except Exception as e:
    #     print(e)
    #     print('Inference session crashed')
    