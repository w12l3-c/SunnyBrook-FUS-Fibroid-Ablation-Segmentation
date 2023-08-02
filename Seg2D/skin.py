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
from Seg2D.save_load import *
from train import deeplabv3_train_model, unet_train_model   

from models import DeepLabV3, Unet, Unetpp


# This will be just segmenting skin (1 class) -> Binary Segmentation Task

# =================== Hyperparameters =================== #
NUM_WORKERS = os.cpu_count() # Number of CPU cores used for data loading
PIN_MEMORY = True   # Pin memory for faster GPU transfer
NUM_EPOCHS = 300 # Just fot test, in pratical should be 100 or more
BATCH_SIZE = 8  # Between 8-16 is good
IN_CHANNELS = 3 # RGB
NUM_CLASSES = 1 # Classes to Segment
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

# =================== Skin Datasets =================== #
# Grab Each Patient's skin image and mask pair
train_skin_dataset = get_skin(train_patients)
test_skin_dataset = get_skin(test_patients)
train_skin_dataset, val_skin_dataset = train_test_split(train_skin_dataset, test_size=0.1, random_state=42)

# Convert the pydicom and mask jpg to PIL images
train_skin_dataset = convert_to_PIL(train_skin_dataset)
val_skin_dataset = convert_to_PIL(val_skin_dataset)
test_skin_dataset = convert_to_PIL(test_skin_dataset)

# ------------------- Training ------------------- #
def deeplabv3_run():
    # Prepare model, transformation, loss function, optimizer, scheduler
    model, transform, loss_fn, optimizer, scheduler = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
    model = model.to(device)
    
    # Prepare train and test dataloader
    if len(train_skin_dataset)%BATCH_SIZE == 1 or len(val_skin_dataset)%BATCH_SIZE == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        val_dataloader = create_dataloader(val_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    else:
        train_dataloader = create_dataloader(train_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
        val_dataloader = create_dataloader(val_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter('runs/DeepLabV3')
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = deeplabv3_train_model(model, train_dataloader, val_dataloader, loss_fn, DeepLabV3.binary_segmentation_iou, optimizer, scheduler, device, NUM_EPOCHS, writer)
    
    print(f"DeepLab v3 Training time: {time.time() - start_time}s")
    print('Training Completed')
    
    # Save the model
    save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/DeepLabV3_Skin_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")


def deeplabv3_inference():
    # Prepare dataset
    dataset = test_skin_dataset
    
    # Load model
    model, transform, loss_fn, optimizer, scheduler = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
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
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = Unet.auto_UNET(in_channels=3, num_classes=2)
    model = model.to(device)
    
    transform = Unet.prepare_transform()
    loss_fn = Unet.prepare_loss('BCE')
    optimizer = Unet.prepare_optimizer(model)
    scheduler = Unet.prepare_scheduler(optimizer)
    
    # Prepare train and test dataloader
    if len(train_skin_dataset)%BATCH_SIZE == 1 or len(val_skin_dataset)%BATCH_SIZE == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter('runs/Unet')
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, Unet.accuracy_iou, optimizer, scheduler, Unet.calculate_weights, device, NUM_EPOCHS, writer)
    
    print(f"UNet Training time: {(time.time() - start_time):.2f}s")
    print('Training Completed')
    
    # Save the model
    save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Skin_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    
    
def unet_inference():
    # Set Seed
    torch.manual_seed(42)
    
    saved_model = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Skin_2023-08-01_18:02:56.pth"
    model = Unet.auto_UNET(in_channels=3, num_classes=2)
    model = load_model_torch(model, saved_model)
    model = model.to(device)
    
    # Run Inference function
    Unet.predict_UNET(model, test_skin_dataset, device)
    
    
def unetpp_run(): 
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = Unetpp.auto_UNETPP(in_channels=3, num_classes=2)
    model = model.to(device)
    
    transform = Unetpp.prepare_transform()
    loss_fn = Unetpp.prepare_loss('BCE')
    optimizer = Unetpp.prepare_optimizer(model)
    scheduler = Unetpp.prepare_scheduler(optimizer)
    
    # Prepare train and test dataloader
    if len(train_skin_dataset)%BATCH_SIZE == 1 or len(val_skin_dataset)%BATCH_SIZE == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_skin_dataset, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter('runs/Unetpp')
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, Unetpp.accuracy_iou, optimizer, scheduler, Unetpp.calculate_weights, device, NUM_EPOCHS, writer)
    
    print(f"UNet++ Training time: {(time.time() - start_time):.2f}s")
    print('Training Completed')
    
    # Save the model
    save_path = f"/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unetpp_Skin_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.pth"
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    
    
def unetpp_inference():
    # Set Seed
    torch.manual_seed(42)
    
    saved_model = ""
    model = Unetpp.auto_UNETPP(in_channels=3, num_classes=2)
    model = load_model_torch(model, saved_model)
    model = model.to(device)
    
    # Run Inference function
    Unetpp.predict_UNET(model, test_skin_dataset, device)
    

if __name__ == "__main__":
    # deeplabv3_run()
    # deeplabv3_inference()
    
    unet_run()
    # unet_inference()
    
    unetpp_run()
    # unetpp_inference()
    
    