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

from dataloader import *
from save_load_2D import *
from train import deeplabv3_train_model

from models import DeepLabV3


# This will be just segmenting spine (1 class) -> Binary Segmentation Task

# ------------------- Hyperparameters ------------------- #
NUM_WORKERS = os.cpu_count() # Number of CPU cores used for data loading
PIN_MEMORY = True   # Pin memory for faster GPU transfer
NUM_EPOCHS = 2 # Just fot test, in pratical should be 100 or more
BATCH_SIZE = 8  # Between 8-16 is good
IN_CHANNELS = 3 # RGB
NUM_CLASSES = 1 # Classes to Segment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def deeplabv3_run():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Define which directories are part of the dataset
    train_path_list = listsiemens[:9]
    test_path_list = [listsiemens[-1]]  # Has to be list format

    # Create a list of patient class objects
    train_patients = create_patient_list(path_list=train_path_list)
    test_patients = create_patient_list(path_list=test_path_list)

    # Grab Each Patient's Spine image and mask pair
    train_spine_dataset = get_spine(train_patients)
    test_spine_dataset = get_spine(test_patients)
    train_spine_dataset, val_spine_dataset = train_test_split(train_spine_dataset, test_size=0.1, random_state=42)
    
    # Convert the pydicom and mask jpg to PIL images
    train_spine_dataset = convert_to_PIL(train_spine_dataset)
    val_spine_dataset = convert_to_PIL(val_spine_dataset)
    test_spine_dataset = convert_to_PIL(test_spine_dataset)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model, transform, loss_fn, optimizer, scheduler = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
    model = model.to(device)
    
    # Prepare train and test dataloader
    if len(train_spine_dataset)%BATCH_SIZE == 1 or len(val_spine_dataset)%BATCH_SIZE == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        val_dataloader = create_dataloader(val_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    else:
        train_dataloader = create_dataloader(train_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
        val_dataloader = create_dataloader(val_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter('runs/DeepLabV3')
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = deeplabv3_train_model(model, train_dataloader, val_dataloader, loss_fn, DeepLabV3.binary_segmentation_iou, optimizer, scheduler, device, NUM_EPOCHS, writer)
    
    print(f"Training time: {time.time() - start_time}s")
    print('Training Actually Worked YAYYYYYYYYYYYYYYYYYYY')
    
    # Save the model
    save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
    save_model_torch(best, save_path)
    
    return test_spine_dataset
    

def deeplabv3_inference():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Patient
    # Define which directories are part of the dataset
    train_path_list = listsiemens[:9]
    test_path_list = [listsiemens[-1]]  # Has to be list format

    # Create a list of patient class objects
    train_patients = create_patient_list(path_list=train_path_list)
    test_patients = create_patient_list(path_list=test_path_list)

    # Grab Each Patient's Spine image and mask pair
    train_spine_dataset = get_spine(train_patients)
    test_spine_dataset = get_spine(test_patients)
    train_spine_dataset, val_spine_dataset = train_test_split(train_spine_dataset, test_size=0.1, random_state=42)
    
    # Convert the pydicom and mask jpg to PIL images
    train_spine_dataset = convert_to_PIL(train_spine_dataset)
    val_spine_dataset = convert_to_PIL(val_spine_dataset)
    test_spine_dataset = convert_to_PIL(test_spine_dataset)
    
    dataset = test_spine_dataset
    
    # Load model
    model = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
    model = load_model_torch(model, "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3_2021-09-22_16:11:39.pth")
    
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
    
    
    
def unet_run():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Define which directories are part of the dataset
    train_path_list = listsiemens[:9]
    test_path_list = [listsiemens[-1]]  # Has to be list format

    # Create a list of patient class objects
    train_patients = create_patient_list(path_list=train_path_list)
    test_patients = create_patient_list(path_list=test_path_list)

    # Grab Each Patient's Spine image and mask pair
    train_spine_dataset = get_spine(train_patients)
    test_spine_dataset = get_spine(test_patients)
    train_spine_dataset, val_spine_dataset = train_test_split(train_spine_dataset, test_size=0.1, random_state=42)
    
    # Convert the pydicom and mask jpg to PIL images
    train_spine_dataset = convert_to_PIL(train_spine_dataset)
    val_spine_dataset = convert_to_PIL(val_spine_dataset)
    test_spine_dataset = convert_to_PIL(test_spine_dataset)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model, transform, loss_fn, optimizer, scheduler = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
    model = model.to(device)
    
    # Prepare train and test dataloader
    if len(train_spine_dataset)%BATCH_SIZE == 1 or len(val_spine_dataset)%BATCH_SIZE == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        val_dataloader = create_dataloader(val_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    else:
        train_dataloader = create_dataloader(train_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
        val_dataloader = create_dataloader(val_spine_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter('runs/DeepLabV3')
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = deeplabv3_train_model(model, train_dataloader, val_dataloader, loss_fn, DeepLabV3.binary_segmentation_iou, optimizer, scheduler, device, NUM_EPOCHS, writer)
    
    print(f"Training time: {time.time() - start_time}s")
    print('Training Actually Worked YAYYYYYYYYYYYYYYYYYYY')
    
    # # Save the model
    save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
    save_model_torch(best, save_path)
    
if __name__ == "__main__":
    # deeplabv3_run()
    deeplabv3_inference()
    