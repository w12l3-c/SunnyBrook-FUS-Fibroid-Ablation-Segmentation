# =============================================================================
# File Description:
# ------------------
# This file contains the training loops, which consists of 
#   1. Training step
#   2. Validation step
#   3. Main looping that called 1 and 2
# =============================================================================

# ==================== Imports ==================== #
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.metrics as metrics

from transformers import Trainer, TrainingArguments

from tqdm.auto import tqdm

# ==================== Early Stopping ==================== #
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """
        Initialize an Early Stopper instance.

        Args:
            patience (int): The number of epochs with no improvement after which to stop training.
            min_delta (float): The minimum change in validation loss required to be considered an improvement.
        """
        self.patience = patience    # Number of epochs with no improvement allowed before early stopping
        self.min_delta = min_delta  # Minimum change in validation loss for improvement to be considered
        self.counter = 0            # Counter to keep track of epochs with no improvement
        self.min_validation_loss = np.inf  # Initialize with positive infinity to track the lowest validation loss

    def early_stop(self, validation_loss):
        """
        Check if early stopping criteria are met.
        Return True if early stopping is needed, otherwise False.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if early stopping criteria are met, otherwise False.
        """
        if validation_loss < self.min_validation_loss:
            # If the current validation loss is lower than the minimum recorded so far
            self.min_validation_loss = validation_loss 
            self.counter = 0  # Reset the counter as there's an improvement
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # If the current validation loss increases by more than min_delta
            self.counter += 1  # Increment the counter to track epochs with no improvement
            if self.counter >= self.patience:
                # If the counter exceeds the patience threshold, early stopping is needed
                return True  
    

# ==================== For DeepLabV3 Only ==================== #
# ==================== Training Step ==================== #
def deeplabv3_train_step(model, dataloader, loss_fn, accuracy, optimizer, scheduler, device):
    """
    Perform a training step for DeepLabV3 model.

    Args:
        model (torch.nn.Module): The DeepLabV3 model.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device (str): Device to perform training on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Training loss and accuracy for the current step.
    """
    model.train()   # Training mode - Gradients are changable
    train_loss = 0
    train_acc = 0

    # Loop over each batch
    for batch, (img, mask) in enumerate(dataloader):
        # Push tensor to gpu or cpu
        img = img.to(device)
        mask = mask.to(device)

        # Forward pass
        y_logits = model(img)['out']
        y_pred = torch.sigmoid(y_logits)
        # y_pred = y_logits.argmax(1).unsqueeze(1)  # Do not do argmax with 1 channel

        # Compute loss and accuracy
        acc = accuracy(y_pred, mask)
        loss = loss_fn(y_logits, mask)

        train_acc += acc
        train_loss += loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % int(len(dataloader)*0.2) == 0 and batch != 0:
            print(f"Batch: {batch}/{len(dataloader)} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    # Total loss and accuracy for the epoch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    # Update the learning rate with scheduler
    scheduler.step()
    
    # print(f"Dice loss: {train_loss:.4f}| Train acc: {train_acc:.4f}")
    return train_loss, train_acc


# ==================== Validation Step ==================== #
def deeplabv3_val_step(model, dataloader, loss_fn, accuracy, device):
    """
    Perform a validation step for DeepLabV3 model.

    Args:
        model (torch.nn.Module): The DeepLabV3 model.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        device (str): Device to perform validation on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Validation loss and accuracy for the current step.
    """
    model.eval()    # Evaluation mode - Gradients are not changable
    val_loss = 0
    val_acc = 0

    # Inferencing without gradient calculation
    with torch.inference_mode():
        # Loop over each batch
        for batch, (img, mask) in enumerate(dataloader):
            # Push tensor to gpu or cpu
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            y_logits = model(img)['out']
            y_pred = torch.sigmoid(y_logits)
            # y_pred = y_logits.argmax(1).unsqueeze(1) 

            # Compute loss and accuracy
            acc = accuracy(y_pred, mask)
            loss = loss_fn(y_logits, mask)

            val_acc += acc
            val_loss += loss
            
            if batch % int(len(dataloader)*0.2) == 0 and batch != 0:
                print(f"Batch: {batch}/{len(dataloader)} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # Total loss and accuracy for the epoch
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    return val_loss, val_acc


# ==================== Training Loop ==================== #
def deeplabv3_train_model(model, train_dataloader, val_dataloader, loss_fn, accuracy, optimizer, scheduler, device, epochs=10, writer=None):
    """
    Train the DeepLabV3 model.

    Args:
        model (torch.nn.Module): The DeepLabV3 model.
        train_dataloader (torch.utils.data.DataLoader): Training data loader.
        val_dataloader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device (str): Device to perform training on (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        writer: Tensorboard SummaryWriter for logging (optional).

    Returns:
        Tuple[dict, dict]: Best model state dict and training results (loss and accuracy).
    """
    # Save results in a dictionary
    results = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_model = deepcopy(model.state_dict())

    # Loop over epochs
    for epoch in tqdm(range(epochs)):
        # Run train anf validation steps
        train_loss, train_acc = deeplabv3_train_step(model, train_dataloader, loss_fn, accuracy, optimizer, scheduler, device)
        val_loss, val_acc = deeplabv3_val_step(model, val_dataloader, loss_fn, accuracy, device)

        # Save results for the epoch
        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss.item())
        results["val_acc"].append(val_acc)

        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # Save the best model
        if train_acc > best_acc:
            best_model = deepcopy(model.state_dict())

        # Tensorboard Tracking
        if writer:
            writer.add_scalar(tag="Loss/train_loss", scalar_value=train_loss.item(), global_step=epoch)
            writer.add_scalar(tag="Loss/val_loss", scalar_value=val_loss.item(), global_step=epoch)

            writer.add_scalar(tag="Accuracy/train_acc", scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag="Accuracy/val_acc", scalar_value=val_acc, global_step=epoch)

            # Track the PyTorch model architecture
            # writer.add_graph(model=model, input_to_model=torch.randn(1, 3, 320, 320).to(device)) # Pass in an example input
    if writer:
        writer.close()

    return best_model, results


# ==================== For UNet and Other Models Only ==================== #
# ==================== Training Step ==================== #
def unet_train_step(model, dataloader, loss_fn, accuracy, optimizer, scheduler, weight_fn, device):
    """
    Perform a training step.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        weight_fn: Weight function.
        device (str): Device to perform training on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Training loss and accuracy for the current step.
    """
    model.train()   # Training mode - Gradients are changable
    train_loss = 0
    train_acc = 0

    # Loop over each batch
    for batch, (img, mask) in enumerate(dataloader):
        # Push tensor to gpu or cpu
        img = img.to(device)
        mask = mask.to(device)
        mask = mask.squeeze()   # Change mask shape for accuarcy and loss function

        # Compute class weight for loss function
        pos_weight = weight_fn(mask).to(device)

        # Forward pass    
        y_logits = model(img)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1).float()

        # Compute loss and accuracy
        acc = accuracy(y_pred, mask)
        loss = loss_fn(y_logits[:, 1], mask) * pos_weight

        train_acc += acc.item()
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % int(len(dataloader)*0.2) == 0 and batch != 0:
            print(f"Progress: {batch}/{len(dataloader)} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    # Compute loss and accuracy for the epoch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    # Update learning rate with scheduler
    scheduler.step(train_loss)

    return train_loss, train_acc


# ==================== Validation Step ==================== #
def unet_val_step(model, dataloader, loss_fn, accuracy, weight_fn, device):
    """
    Perform a validation step.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        weight_fn: Weight function.
        device (str): Device to perform validation on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Validation loss and accuracy for the current step.
    """
    model.eval()    # Evaluation mode - Gradients are not changable
    val_loss = 0
    val_acc = 0

    # Inference
    with torch.inference_mode():    
        # Loop over each batch  
        for batch, (img, mask) in enumerate(dataloader):
            # Push tensor to gpu or cpu
            img = img.to(device)
            mask = mask.to(device)
            mask = mask.squeeze()   # Change mask shape for accuarcy and loss function

            # Compute class weight for loss function
            pos_weight = weight_fn(mask).to(device)

            # Forward pass
            y_logits = model(img)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1).float()

            # Compute loss and accuracy
            acc = accuracy(y_pred, mask)
            loss = loss_fn(y_logits[:, 1], mask) * pos_weight

            val_acc += acc.item()
            val_loss += loss.item()
            
            if batch % 4 == 0 and batch != 0:
                print(f"Progress: {batch}/{len(dataloader)} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # Compute loss and accuracy for the epoch
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    return val_loss, val_acc


# ==================== Training Loop ==================== #
def unet_train_model(model, train_dataloader, val_dataloader, loss_fn, accuracy, optimizer, scheduler, weight_fn, device, epochs=10, writer=None):
    """
    Current this method can be used for Unet, Unet++, DeepLabv3+
    Training the model while storing the loss and accuracy in tensorboard

    Args:
        model (torch.nn.Module): The UNet model.
        train_dataloader (torch.utils.data.DataLoader): Training data loader.
        val_dataloader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        weight_fn: Weight function.
        device (str): Device to perform training on (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        writer: Tensorboard SummaryWriter for logging (optional).

    Returns:
        Tuple[dict, dict]: Best model state dict and training results (loss and accuracy).
    """
    # Initialize results in a dictionary
    results = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Initialize early stopper
    early_stopper = EarlyStopper(patience=5, min_delta=5)

    # Training
    for epoch in tqdm(range(epochs)):
        # Run training and validation steps
        train_loss, train_acc = unet_train_step(model, train_dataloader, loss_fn, accuracy, optimizer, scheduler, weight_fn, device)
        val_loss, val_acc = unet_val_step(model, val_dataloader, loss_fn, accuracy, weight_fn, device)

        # Update the current stats of the epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # Tensorboard Tracking
        if writer:
            writer.add_scalar(tag="Loss/train_loss", scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag="Loss/val_loss", scalar_value=val_loss, global_step=epoch)

            writer.add_scalar(tag="Accuracy/train_acc", scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag="Accuracy/val_acc", scalar_value=val_acc, global_step=epoch)

            # Track the PyTorch model architecture
            # writer.add_graph(model=model, input_to_model=torch.randn(1, 3, 320, 320).to(device)) # Pass in an example input

        # Early stopping
        if early_stopper.early_stop(val_loss):
            print('Model has not improved in 15 epochs.') 
            print('Early stopping.........')            
            break
        
    if writer:
        writer.close()

    # Save the best model
    best_model = deepcopy(model.state_dict())

    return best_model, results

# ==================== Multiclass ==================== #
# ==================== Training Step ==================== #
def train_step(model, dataloader, loss_fn, accuracy, optimizer, scheduler, weight_fn, device):
    """
    Perform a training step.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        weight_fn: Weight function.
        device (str): Device to perform training on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Training loss and accuracy for the current step.
    """
    model.train()
    train_loss = 0
    train_acc = 0

    for batch, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)

        y_logits = model(img)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1).float()

        acc = accuracy(y_pred, mask)
        
        y_logits = y_logits.permute(0, 2, 3, 1)
        y_logits = y_logits.view(-1, 8)
        mask = mask.view(-1)
        
        loss = loss_fn(y_logits, mask)

        train_acc += acc.item()
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % int(len(dataloader)*0.2) == 0 and batch != 0:
            print(f"Progress: {batch}/{len(dataloader)} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    scheduler.step(train_loss)

    return train_loss, train_acc


# ==================== Validation Step ==================== #
def val_step(model, dataloader, loss_fn, accuracy, weight_fn, device):
    """
    Perform a validation step.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        weight_fn: Weight function.
        device (str): Device to perform validation on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Validation loss and accuracy for the current step.
    """
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.inference_mode():
        for batch, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)
            mask = mask.squeeze()

            pos_weight = weight_fn(mask).to(device)

            y_logits = model(img)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1).float()

            acc = accuracy(y_pred, mask)
        
            y_logits = y_logits.permute(0, 2, 3, 1)
            y_logits = y_logits.view(-1, 8)
            mask = mask.view(-1)
            
            loss = loss_fn(y_logits, mask)

            val_acc += acc.item()
            val_loss += loss.item()
            
            if batch % 4 == 0 and batch != 0:
                print(f"Progress: {batch}/{len(dataloader)} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    return val_loss, val_acc


# ==================== Training Loop ==================== #
def train_model(model, train_dataloader, val_dataloader, loss_fn, accuracy, optimizer, scheduler, weight_fn, device, epochs=10, writer=None):
    """
    Multi-class training.
    
    Args:
        model (torch.nn.Module): The UNet model.
        train_dataloader (torch.utils.data.DataLoader): Training data loader.
        val_dataloader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: The loss function.
        accuracy: The accuracy function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        weight_fn: Weight function.
        device (str): Device to perform training on (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        writer: Tensorboard SummaryWriter for logging (optional).

    Returns:
        Tuple[dict, dict]: Best model state dict and training results (loss and accuracy).
    """
        # Initialize results in a dictionary
    results = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Initialize early stopper
    early_stopper = EarlyStopper(patience=5, min_delta=5)

    # Training
    for epoch in tqdm(range(epochs)):
        # Run training and validation steps
        train_loss, train_acc = unet_train_step(model, train_dataloader, loss_fn, accuracy, optimizer, scheduler, weight_fn, device)
        val_loss, val_acc = unet_val_step(model, val_dataloader, loss_fn, accuracy, weight_fn, device)

        # Update the current stats of the epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # Tensorboard Tracking
        if writer:
            writer.add_scalar(tag="Loss/train_loss", scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag="Loss/val_loss", scalar_value=val_loss, global_step=epoch)

            writer.add_scalar(tag="Accuracy/train_acc", scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag="Accuracy/val_acc", scalar_value=val_acc, global_step=epoch)

            # Track the PyTorch model architecture
            # writer.add_graph(model=model, input_to_model=torch.randn(1, 3, 320, 320).to(device)) # Pass in an example input

        # Early stopping
        if early_stopper.early_stop(val_loss):
            print('Model has not improved in 15 epochs.') 
            print('Early stopping.........')            
            break
        
    if writer:
        writer.close()

    # Save the best model
    best_model = deepcopy(model.state_dict())

    return best_model, results

# ==================== For Beit3 Only ==================== #
# Need testing first
def beit3_train_model(model, dataset_dict, epochs, batch_size, compute_metrics, push_to_hub=False):
    """
    Train the Beit3 model.

    Args:
        model: The Beit3 model.
        dataset_dict (dict): Dictionary containing training and validation datasets.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        compute_metrics: Function to compute metrics.
        push_to_hub (bool): Whether to push the model to the Hugging Face Hub (optional).
    """
    training_args = TrainingArguments(
        output_dir="/mnt/HDD_1TB/Wallace/Code/Seg2D/predictions",
        # learning_rate=1e-3,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # save_total_limit=3,
        # evaluation_strategy="steps",
        save_strategy="epochs",
        # save_steps=20,
        # eval_steps=20,
        # logging_steps=1,
        # eval_accumulation_steps=5,
        # remove_unused_columns=False,
        # push_to_hub=push_to_hub,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        compute_metrics=compute_metrics,
    )
    
    trainer.train()