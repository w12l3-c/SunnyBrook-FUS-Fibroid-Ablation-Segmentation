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
from model_run_fn import deeplabv3_run, unet_run, unetpp_run, deeplabv3p_run, manet_run, fpn_run
from model_inference_fn import deeplabv3_inference, unet_inference, unetpp_inference, deeplabv3plus_inference

import bowel, spine, skin


# This will be just segmenting muscle (1 class) -> Binary Segmentation Task

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
train_path_list = listsiemens[:7] + listsiemens[8:] 
test_path_list = [listsiemens[7]]  

# Create a list of patient class objects
train_patients = create_patient_list(path_list=train_path_list)
test_patients = create_patient_list(path_list=test_path_list)

# =================== Muscle Datasets =================== #
# Grab Each Patient's muscle image and mask pair
train_muscle_dataset = get_muscle(train_patients)
test_muscle_dataset = get_muscle(test_patients)
train_muscle_dataset, val_muscle_dataset = train_test_split(train_muscle_dataset, test_size=0.1, random_state=42)

# Convert the pydicom and mask jpg to PIL images
train_muscle_dataset = convert_to_PIL(train_muscle_dataset)
val_muscle_dataset = convert_to_PIL(val_muscle_dataset)
test_muscle_dataset = convert_to_PIL(test_muscle_dataset)

if __name__ == "__main__":

    # unet_writer = f"runs/Unet_Muscle_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    # unet_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Muscle_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    # unet_run(train_muscle_dataset, val_muscle_dataset, device, NUM_EPOCHS, 8, NUM_WORKERS, True, unet_writer, unet_save_path, 'BCE')
    unet_save_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Muscle_2023-08-10_11.pth"
    unet_inference(test_muscle_dataset, unet_save_path, device)
    
    # Trash
    # try:
    #     # unetpp_writer = f"runs/UnetPP_Muscle_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    #     # unetpp_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/UnetP_Muscle_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    #     # unetpp_run(train_muscle_dataset, val_muscle_dataset, device, NUM_EPOCHS, 4, NUM_WORKERS, True, unetpp_writer, unetpp_save_path, 'BCE')
    #     unetpp_save_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/UnetPP_Muscle_2023-08-04_16.pth"
    #     unetpp_inference(test_muscle_dataset, unetpp_save_path, device)
    # except Exception as e:
    #     print(e)
    #     print('This training sessions failed')

    # Model ok Compare with Unet later
    try:
        # deeplabv3plus_writer = f"runs/DeepLabV3P_Muscle_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
        # deeplabv3plus_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3P_Muscle_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
        # deeplabv3p_run(train_muscle_dataset, val_muscle_dataset, device, NUM_EPOCHS, 4, NUM_WORKERS, True, deeplabv3plus_writer, deeplabv3plus_save_path, 'BCE')
        deeplabv3plus_save_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3P_Muscle_2023-08-05_00.pth"
        deeplabv3plus_inference(test_muscle_dataset, deeplabv3plus_save_path, device)
    except Exception as e:
        print(e)
        print('This training sessions failed')
    
    
        
    # No model
    # try:
    #     manet_writer = f"runs/MAnet_Spine_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    #     manet_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/MAnet_Spine_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    #     manet_run(spine.train_spine_dataset, spine.val_spine_dataset, device, NUM_EPOCHS, 4, NUM_WORKERS, True, manet_writer, manet_save_path, 'BCE')
    # except Exception as e:
    #     print(e)
    #     print('This training sessions failed')
    

    # Overfitting I think
    # try:
    #     # unet_writer = f"runs/Unet_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    #     # unet_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    #     # unet_run(bowel.train_bowel_dataset, bowel.val_bowel_dataset, device, NUM_EPOCHS, 8, NUM_WORKERS, True, unet_writer, unet_save_path, 'BCE')
    #     unet_save_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Bowel_2023-08-05_08.pth"
    #     unet_inference(bowel.val_bowel_dataset[:15], unet_save_path, device)
    # except Exception as e:
    #     print(e)
    #     print('This training sessions failed')
    
    # Trash
    # try:
    #     # unetpp_writer = f"runs/UnetPP_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}"
    #     # unetpp_save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/UnetPP_Bowel_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    #     # unetpp_run(bowel.train_bowel_dataset, bowel.val_bowel_dataset, device, NUM_EPOCHS, 4, NUM_WORKERS, True, unetpp_writer, unetpp_save_path, 'BCE')
    #     unetpp_save_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/UnetPP_Muscle_2023-08-04_16.pth"
    #     unetpp_inference(bowel.val_bowel_dataset[:15], unetpp_save_path, device)
    # except Exception as e:
    #     print('This training sessions failed')