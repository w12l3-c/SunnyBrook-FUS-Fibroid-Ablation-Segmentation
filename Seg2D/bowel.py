# ======================================================================== #
# File Description:
# ------------------
# This file is used for train or inferencing model on the Bowel Dataset
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

import skin, hipl, hipr

# This will be just segmenting bowel (1 class) -> Binary Segmentation Task

# =================== Hyperparameters =================== #
NUM_WORKERS = os.cpu_count() # Number of CPU cores used for data loading
PIN_MEMORY = True   # Pin memory for faster GPU transfer
NUM_EPOCHS = 200 # Just fot test, in pratical should be 100 or more
BATCH_SIZE = 8  # Between 8-16 is good
IN_CHANNELS = 3 # RGB
NUM_CLASSES = 2 # Classes to Segment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================== Patient Dataset =================== #
# Set seed for reproducibility
torch.manual_seed(42)

# Define which directories are part of the dataset
train_path_list = listsiemens[:7] + listsiemens[8:] 
test_path_list = [listsiemens[7]]  

# Create a list of patient class objects
train_patients = create_patient_list(path_list=train_path_list)
test_patients = create_patient_list(path_list=test_path_list)

# =================== Bowel Datasets =================== #
# Grab Each Patient's bowel image and mask pair
train_bowel_dataset = get_bowel(train_patients)
test_bowel_dataset = get_bowel(test_patients)
train_bowel_dataset, val_bowel_dataset = train_test_split(train_bowel_dataset, test_size=0.1, random_state=42)

# Convert the pydicom and mask jpg to PIL images
train_bowel_dataset = convert_to_PIL(train_bowel_dataset)
val_bowel_dataset = convert_to_PIL(val_bowel_dataset)
test_bowel_dataset = convert_to_PIL(test_bowel_dataset)

if __name__ == "__main__":  
    # =================== Training & Inference =================== #
    try:
        # Set the tensorboard and save path
        unet_writer = f"runs/Unet_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
        unet_save_path = f"./trained_models/Unet_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
        unet_run(train_bowel_dataset, val_bowel_dataset, device, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, True, unet_writer, unet_save_path, 'BCE')
        # unet_save_path = "./trained_models/Unet_Bowel_2023-08-05_08.pth"
        #   # display=True to show individual images and results, display=False to show the overall stats of entire dataset
        # unet_inference(test_bowel_dataset, unet_save_path, device)
    except Exception as e:
        print(e)
        print('This training sessions failed')
    
    # =================== Training & Inference =================== #
    try:
        #   # Set the tensorboard and save path
        # unetpp_writer = f"runs/UnetPP_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
        # unetpp_save_path = f"./trained_models/UnetPP_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
        # unetpp_run(train_bowel_dataset, val_bowel_dataset, device, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, True, unetpp_writer, unetpp_save_path, 'BCE')
        unetpp_save_path = "./trained_models/UnetPP_Muscle_2023-08-04_16.pth"
        # display=True to show individual images and results, display=False to show the overall stats of entire dataset
        unetpp_inference(test_bowel_dataset, unetpp_save_path, device)
    except Exception as e:
        print('This training sessions failed')
    
    # =================== Training =================== #
    try:
        # Set the tensorboard and save path
        deeplabv3p_writer = f"runs/DeepLabV3P_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
        deeplabv3p_save_path = f"./trained_models/DeepLabV3P_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
        deeplabv3p_run(train_bowel_dataset, val_bowel_dataset, device, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, True, deeplabv3p_writer, deeplabv3p_save_path, 'BCE')
    except Exception as e:
        print(e)
        print('This training sessions failed')
        
    
    
    
        
    