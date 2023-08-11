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

# =================== Run Train Script =================== #
# All of these functions are for training individual models
# The scipt with the parts name ex. spine.py will run these functions 


def deeplabv3_run(train_dataset, val_dataset, device, epochs, batch_size, num_workers, pin_memory, writer_path, save_path):
    """
    Train the DeepLabV3 model and save the best model checkpoint.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        device (torch.device): Device for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.
        writer_path (str): Path for TensorBoard writer.
        save_path (str): Path to save the trained model checkpoint.
    """
    # Prepare model, transformation, loss function, optimizer, scheduler
    model, transform, loss_fn, optimizer, scheduler = DeepLabV3.DeepLabV3(in_channels=3, num_classes=1, size='regular')
    model = model.to(device)
    
    # Prepare train and test dataloader
    if len(train_dataset)%batch_size == 1 or len(val_dataset)%batch_size == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    else:
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter(writer_path)
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = deeplabv3_train_model(model, train_dataloader, val_dataloader, loss_fn, DeepLabV3.binary_segmentation_iou, optimizer, scheduler, device, epochs, writer)
    
    print(f"DeepLabV3 Training time: {((time.time() - start_time)/60/60):.2f}hrs")
    print('Training Completed')
    
    # Save the model
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    
    
def unet_run(train_dataset, val_dataset, device, epochs, batch_size, num_workers, pin_memory, writer_path, save_path, loss='BCE', axis='Sagittal'): 
    """
    Train the UNet model and save the best model checkpoint.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        device (torch.device): Device for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.
        writer_path (str): Path for TensorBoard writer.
        save_path (str): Path to save the trained model checkpoint.
        loss (str, optional): Loss function to use (default is 'BCE').
        axis (str, optional): Axis for flip augmentation (default is 'Sagittal').
    """
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = Unet.auto_UNET(in_channels=3, num_classes=2)
    model = model.to(device)
    
    flip = 0.3 if axis == 'Sagittal' else 0.0
    transform = Unet.prepare_transform(flip)
    loss_fn = Unet.prepare_loss(loss)
    optimizer = Unet.prepare_optimizer(model, option='AdamW')
    scheduler = Unet.prepare_scheduler(optimizer)

    # Prepare train and test dataloader
    if len(train_dataset)%batch_size == 1 or len(val_dataset)%batch_size == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter(writer_path)
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, Unet.accuracy_iou, optimizer, scheduler, Unet.calculate_weights, device, epochs, writer)
    
    print(f"UNet Training time: {((time.time() - start_time)/60/60):.2f}hrs")
    print('Training Completed')
    
    # Save the model
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    

def unetpp_run(train_dataset, val_dataset, device, epochs, batch_size, num_workers, pin_memory, writer_path, save_path, loss='BCE', axis='Sagittal'): 
    """
    Train the UNet++ model and save the best model checkpoint.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        device (torch.device): Device for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.
        writer_path (str): Path for TensorBoard writer.
        save_path (str): Path to save the trained model checkpoint.
        loss (str, optional): Loss function to use (default is 'BCE').
        axis (str, optional): Axis for flip augmentation (default is 'Sagittal').
    """
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = Unetpp.auto_UNETPP(in_channels=3, num_classes=2)
    model = model.to(device)
    
    flip = 0.3 if axis == 'Sagittal' else 0.0
    transform = Unetpp.prepare_transform(flip)
    loss_fn = Unetpp.prepare_loss(loss)
    optimizer = Unetpp.prepare_optimizer(model, option='AdamW')
    scheduler = Unetpp.prepare_scheduler(optimizer)
    
    # Prepare train and test dataloader
    if len(train_dataset)%batch_size == 1 or len(val_dataset)%batch_size == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter(writer_path)
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, Unetpp.accuracy_iou, optimizer, scheduler, Unetpp.calculate_weights, device, epochs, writer)
    
    print(f"UNet++ Training time: {((time.time() - start_time)/60/60):.2f}hrs")
    print('Training Completed')
    
    # Save the model
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    
    
def deeplabv3p_run(train_dataset, val_dataset, device, epochs, batch_size, num_workers, pin_memory, writer_path, save_path, loss='BCE', axis='Sagittal'): 
    """
    Train the DeepLabV3+ model and save the best model checkpoint.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        device (torch.device): Device for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.
        writer_path (str): Path for TensorBoard writer.
        save_path (str): Path to save the trained model checkpoint.
        loss (str, optional): Loss function to use (default is 'BCE').
        axis (str, optional): Axis for flip augmentation (default is 'Sagittal').
    """
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = DeepLabV3plus.auto_DEEPLABV3P(in_channels=3, num_classes=2)
    model = model.to(device)
    
    flip = 0.3 if axis == 'Sagittal' else 0.0
    transform = DeepLabV3plus.prepare_transform(flip)
    loss_fn = DeepLabV3plus.prepare_loss(loss)
    optimizer = DeepLabV3plus.prepare_optimizer(model)
    scheduler = DeepLabV3plus.prepare_scheduler(optimizer, option='AdamW')
    
    # Prepare train and test dataloader
    if len(train_dataset)%batch_size == 1 or len(val_dataset)%batch_size == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter(writer_path)
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, DeepLabV3plus.accuracy_iou, optimizer, scheduler, DeepLabV3plus.calculate_weights, device, epochs, writer)
    
    print(f"DeepLabV3+ Training time: {((time.time() - start_time)/60/60):.2f}hrs")
    print('Training Completed')
    
    # Save the model
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    
    
def fpn_run(train_dataset, val_dataset, device, epochs, batch_size, num_workers, pin_memory, writer_path, save_path, loss='BCE', axis='Sagittal'): 
    """
    Train the FPN model and save the best model checkpoint.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        device (torch.device): Device for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.
        writer_path (str): Path for TensorBoard writer.
        save_path (str): Path to save the trained model checkpoint.
        loss (str, optional): Loss function to use (default is 'BCE').
        axis (str, optional): Axis for flip augmentation (default is 'Sagittal').
    """
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = FPN.auto_FPN(in_channels=3, num_classes=2)
    model = model.to(device)
    
    flip = 0.3 if axis == 'Sagittal' else 0.0
    transform = FPN.prepare_transform(flip)
    loss_fn = FPN.prepare_loss(loss)
    optimizer = FPN.prepare_optimizer(model)
    scheduler = FPN.prepare_scheduler(optimizer, option='AdamW')
    
    # Prepare train and test dataloader
    if len(train_dataset)%batch_size == 1 or len(val_dataset)%batch_size == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter(writer_path)
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, FPN.accuracy_iou, optimizer, scheduler, FPN.calculate_weights, device, epochs, writer)
    
    print(f"FPN Training time: {(time.time() - start_time):.2f}s")
    print('Training Completed')
    
    # Save the model
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    
    
def manet_run(train_dataset, val_dataset, device, epochs, batch_size, num_workers, pin_memory, writer_path, save_path, loss='BCE', axis='Sagittal'): 
    """
    Train the MAnet model and save the best model checkpoint.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        device (torch.device): Device for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.
        writer_path (str): Path for TensorBoard writer.
        save_path (str): Path to save the trained model checkpoint.
        loss (str, optional): Loss function to use (default is 'BCE').
        axis (str, optional): Axis for flip augmentation (default is 'Sagittal').
    """
    # Set Seed
    torch.manual_seed(42)
    
    # Prepare model, transformation, loss function, optimizer, scheduler
    model = MAnet.auto_MANET(in_channels=3, num_classes=2)
    model = model.to(device)
    
    flip = 0.3 if axis == 'Sagittal' else 0.0
    transform = MAnet.prepare_transform(flip)
    loss_fn = MAnet.prepare_loss(loss)
    optimizer = MAnet.prepare_optimizer(model, option='AdamW')
    scheduler = MAnet.prepare_scheduler(optimizer)
    
    # Prepare train and test dataloader
    if len(train_dataset)%batch_size == 1 or len(val_dataset)%batch_size == 1:
        # Batch number of 1 will cause error in batchnorm
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=None)
    else:
        train_dataloader = create_dataloader(train_dataset, transform, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)
        val_dataloader = create_dataloader(val_dataset, transform, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)    
    
    # Prepare tensorboard writer
    writer = SummaryWriter(writer_path)
    
    # Start time
    start_time = time.time()
    
    # Training Loop
    best, results = unet_train_model(model, train_dataloader, val_dataloader, loss_fn, MAnet.accuracy_iou, optimizer, scheduler, MAnet.calculate_weights, device, epochs, writer)
    
    print(f"MAnet Training time: {(time.time() - start_time):.2f}s")
    print('Training Completed')
    
    # Save the model
    save_model_torch(best, save_path)
    print(f"Model saved at {save_path}")
    