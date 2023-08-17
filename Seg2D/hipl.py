# ======================================================================== #
# File Description:
# ------------------
# This file is used for train or inferencing model on the Left Hip Dataset
# It is a binary segmentation task
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
from train import deeplabv3_train_model, unet_train_model   

from models import DeepLabV3, Unet, Unetpp
from model_run_fn import deeplabv3_run, unet_run, unetpp_run, deeplabv3p_run
from model_inference_fn import deeplabv3_inference, unet_inference, unetpp_inference

# =================== Hyperparameters =================== #
# To set hyperparams for model, go to model_run_fn.py 
NUM_WORKERS = os.cpu_count() # Number of CPU cores used for data loading
PIN_MEMORY = True # Pin memory for faster GPU transfer
NUM_EPOCHS = 150 # 100-300
BATCH_SIZE = 8 # Optimally 8-16
IN_CHANNELS = 3 # RGB
NUM_CLASSES = 2 # Classes to Segment
device = 'cuda' if torch.cuda.is_available() else 'cpu' # gpu if available

# =================== Patient Dataset =================== #
# Set seed for reproducibility
torch.manual_seed(42)

# Define which directories are part of the dataset
train_path_list = listsiemens[:7] + listsiemens[8:] + listbones[:-1] + listlowres
test_path_list = [listsiemens[7]] + [listbones[-1]]

# Create a list of patient class objects
train_patients = create_patient_list(path_list=train_path_list)
test_patients = create_patient_list(path_list=test_path_list)

# =================== HipL Datasets =================== #
# Grab Each Patient's HipL image and mask pair
train_hipl_dataset = get_hipl(train_patients)
test_hipl_dataset = get_hipl(test_patients)
train_hipl_dataset, val_hipl_dataset = train_test_split(train_hipl_dataset, test_size=0.1, random_state=42)

# Convert the pydicom and mask jpg to PIL images
train_hipl_dataset = convert_to_PIL(train_hipl_dataset, img_size=(160, 160))
val_hipl_dataset = convert_to_PIL(val_hipl_dataset, img_size=(160, 160))
test_hipl_dataset = convert_to_PIL(test_hipl_dataset, img_size=(160, 160))

if __name__ == "__main__":
    # Training Block
    # try:
    #     # Set up the tensorboard and save path
    #     unet_writer = f"runs/Unet_HipL_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    #     unet_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_HipL_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    #     # Traing Model
    #     unet_run(train_hipl_dataset, val_hipl_dataset, device, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, True, unet_writer, unet_save_path, 'BCE')
    # except Exception as e:
    #     print(e)
    #     print('Training Session Crashed')
        
    # Inference Block
    try:
        # Set up the path where the model '.pth' file is saved
        unet_save_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_HipL_2023-08-10_16.pth"
        # Inference Model
        unet_inference(test_hipl_dataset, unet_save_path, device, img_size=(160, 160), display=False)
    except Exception as e:
        print(e)
        print('Inference Session Crashed')
        
    try:
        deeplabv3p_writer = f"runs/DeepLabV3P_HipL_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
        deeplabv3p_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3P_HipL_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
        deeplabv3p_run(train_hipl_dataset, val_hipl_dataset, device, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, True, deeplabv3p_writer, deeplabv3p_save_path, 'BCE')
    except Exception as e:
        print(e)
        print('This training sessions failed')
    
    