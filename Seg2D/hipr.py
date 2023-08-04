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


# This will be just segmenting hipr (1 class) -> Binary Segmentation Task

# =================== Hyperparameters =================== #
NUM_WORKERS = os.cpu_count() # Number of CPU cores used for data loading
PIN_MEMORY = True   # Pin memory for faster GPU transfer
NUM_EPOCHS = 300 # Just fot test, in pratical should be 100 or more
BATCH_SIZE = 8  # Between 8-16 is good
IN_CHANNELS = 3 # RGB
NUM_CLASSES = 2 # Classes to Segment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================== Patient Dataset =================== #
# Set seed for reproducibility
torch.manual_seed(42)

# Define which directories are part of the dataset
train_path_list = listsiemens[:7] + listsiemens[8:] + listbones[:-1] + listlowres
test_path_list = [listsiemens[7]] + [listbones[-1]]

# Create a list of patient class objects
train_patients = create_patient_list(path_list=train_path_list)
test_patients = create_patient_list(path_list=test_path_list)

# =================== Hip R Datasets =================== #
# Grab Each Patient's Hip R image and mask pair
train_hipr_dataset = get_hipr(train_patients)
test_hipr_dataset = get_hipr(test_patients)
train_hipr_dataset, val_hipr_dataset = train_test_split(train_hipr_dataset, test_size=0.1, random_state=42)

# Convert the pydicom and mask jpg to PIL images
train_hipr_dataset = convert_to_PIL(train_hipr_dataset, img_size=(160, 320))
val_hipr_dataset = convert_to_PIL(val_hipr_dataset, img_size=(160, 320))
test_hipr_dataset = convert_to_PIL(test_hipr_dataset, img_size=(160, 320))

if __name__ == "__main__":
    # deeplabv3_run()
    # deeplabv3_inference()
    
    unet_writer = f"runs/Unet_HipR_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    unet_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_HipR_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    unet_run(train_hipr_dataset, val_hipr_dataset, device, NUM_EPOCHS, 8, NUM_WORKERS, True, unet_writer, unet_save_path, 'BCE')
    # unet_inference()
    
    # unetpp_writer = runs/UnetPP_HipR_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}
    # unetpp_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/UnetPP_HipR_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    # unetpp_run(train_hipr_dataset, val_hipr_dataset, device, NUM_EPOCHS, 4, NUM_WORKERS, True, unetpp_writer, unetpp_save_path, 'BCE')
    # # unetpp_inference()
    
    # deeplabv3plus_writer = runs/DeepLabV3P_HipR_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}
    # deeplabv3plus_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3P_HipR_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    # deeplabv3p_run(train_hipr_dataset, val_hipr_dataset, device, NUM_EPOCHS, 4, NUM_WORKERS, True, deeplabv3plus_writer, deeplabv3plus_save_path, 'BCE')
    # # deeplabv3plus_inference()
    
    