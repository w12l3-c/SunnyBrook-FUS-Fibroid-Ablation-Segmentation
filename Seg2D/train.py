import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.metrics as metrics

from tqdm.auto import tqdm

# ---------------------- For DeepLabV3 Only ---------------------- #
# ---------------------- Training Step ---------------------- #
def deeplabv3_train_step(model, dataloader, loss_fn, accuracy, optimizer, scheduler, device):
    model.train()
    train_loss = 0
    train_acc = 0

    for batch, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)

        y_logits = model(img)['out']
        y_pred = torch.sigmoid(y_logits)
        # y_pred = y_logits.argmax(1).unsqueeze(1)  # Do not do argmax in binary classification

        acc = accuracy(y_pred, mask)
        loss = loss_fn(y_logits, mask)

        train_acc += acc
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 20 == 0:
            print(f"Batch: {batch}/{len(dataloader)} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    scheduler.step()
    # print(f"Dice loss: {train_loss:.4f}| Train acc: {train_acc:.4f}")

    return train_loss, train_acc


# ---------------------- Validation Step ---------------------- #
def deeplabv3_val_step(model, dataloader, loss_fn, accuracy, device):
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.inference_mode():
        for batch, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)

            y_logits = model(img)['out']
            y_pred = torch.sigmoid(y_logits)
            # y_pred = y_logits.argmax(1).unsqueeze(1)  # Do not do argmax in binary classification

            acc = accuracy(y_pred, mask)
            loss = loss_fn(y_logits, mask)

            val_acc += acc
            val_loss += loss
            
            if batch % 5 == 0:
                print(f"Batch: {batch}/{len(dataloader)} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
    # print(f"Dice loss: {val_loss:.4f}| Val acc: {val_acc:.4f}")

    return val_loss, val_acc


# ---------------------- Training Loop ---------------------- #
def deeplabv3_train_model(model, train_dataloader, val_dataloader, loss_fn, accuracy, optimizer, scheduler, device, epochs=10, writer=None):
    results = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_model = deepcopy(model.state_dict())

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = deeplabv3_train_step(model, train_dataloader, loss_fn, accuracy, optimizer, scheduler, device)
        val_loss, val_acc = deeplabv3_val_step(model, val_dataloader, loss_fn, accuracy, device)

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss.item())
        results["val_acc"].append(val_acc)

        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

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


# ---------------------- For UNet Only ---------------------- #
# ---------------------- Training Step ---------------------- #
def train_step(model, dataloader, loss_fn, accuracy, optimizer, scheduler, device):
    model.train()
    train_loss = 0
    train_acc = 0

    for batch, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)

        y_logits = model(img)
        y_pred = torch.sigmoid(y_logits)
        # y_pred = y_logits.argmax(1).unsqueeze(1)  # Do not do argmax in binary classification

        acc = accuracy(y_pred, mask)
        loss = loss_fn(y_logits, mask)

        train_acc += acc
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 20 == 0:
            print(f"Batch: {batch}/{len(dataloader)} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    scheduler.step()
    # print(f"Dice loss: {train_loss:.4f}| Train acc: {train_acc:.4f}")

    return train_loss, train_acc


# ---------------------- Validation Step ---------------------- #
def val_step(model, dataloader, loss_fn, accuracy, device):
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.inference_mode():
        for batch, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)

            y_logits = model(img)
            y_pred = torch.sigmoid(y_logits)
            # y_pred = y_logits.argmax(1).unsqueeze(1)  # Do not do argmax in binary classification

            acc = accuracy(y_pred, mask)
            loss = loss_fn(y_logits, mask)

            val_acc += acc
            val_loss += loss
            
            if batch % 5 == 0:
                print(f"Batch: {batch}/{len(dataloader)} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
    # print(f"Dice loss: {val_loss:.4f}| Val acc: {val_acc:.4f}")

    return val_loss, val_acc


# ---------------------- Training Loop ---------------------- #
def unet_train_model(model, train_dataloader, val_dataloader, loss_fn, accuracy, optimizer, scheduler, device, epochs=10, writer=None):
    results = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_model = deepcopy(model.state_dict())

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, accuracy, optimizer, scheduler, device)
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, accuracy, device)

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss.item())
        results["val_acc"].append(val_acc)

        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

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
