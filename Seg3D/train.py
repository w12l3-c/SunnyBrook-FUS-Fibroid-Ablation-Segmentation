import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from utils import *
from vnet import *

# ---------------------- Training Step ---------------------- #
def train_step(model, dataloader, loss_fn, optimizer, accuracy, device):
    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)
        
        y_logits = model(img)
        y = flatten_mask(mask)
        
        acc = accuracy(y_logits, y)
        loss = loss_fn(y_logits, y)

        train_acc += acc
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
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
            y = flatten_mask(mask)
            
            val_acc += accuracy(y_logits, y)
            val_loss += loss_fn(y_logits, y)
    
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
    # print(f"Dice loss: {val_loss:.4f}| Val acc: {val_acc:.4f}")
    
    return val_loss, val_acc


# ---------------------- Training Loop ---------------------- #
def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, accuracy, device, epochs=10, writer=None):
    results = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy, device)
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, accuracy, device)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")
        
        if train_acc > best_acc:
            best_model = deepcopy(model.state_dict())
        
        # Tensorboard Tracking
        if writer:
            writer.add_scalars(main_tag="Loss", 
                        tag_scalar_dict={"train_loss": train_loss, "test_loss": val_loss},
                        global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", 
                                tag_scalar_dict={"train_acc": train_acc, "test_acc": val_acc}, 
                                global_step=epoch)
            
            # Track the PyTorch model architecture
            writer.add_graph(model=model, input_to_model=torch.randn(1, 1, 32, 32, 32).to(device)) # Pass in an example input
    if writer:
        writer.close()
        
    return best_model, results
    
    
# ---------------------- Plot things rn ---------------------- #
# Update: This doesn't work
def plot_results(results):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for item in results.values():
        for j in item:
          j = j.detach().cpu().numpy()

    ax[0][0].plot(results["train_loss"], label="train_loss")
    ax[0][0].set_title("Train Dice Loss")
    ax[0][1].plot(results["train_acc"], label="train_acc")
    ax[0][1].set_title("Train Accuracy")
    ax[1][0].plot(results["test_loss"], label="test_loss")
    ax[1][0].set_title("Test Dice Loss")
    ax[1][1].plot(results["test_acc"], label="test_acc")
    ax[1][1].set_title("Test Accuracy")
    plt.show()
    
    
# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    example_train_dataloader = [(torch.rand(2, 1, 32, 32, 32), torch.randint(low=0, high=2, size=(2, 1, 32, 32, 32), dtype=torch.int64)) for _ in range(3)]
    example_test_dataloader = [(torch.rand(1, 1, 32, 32, 32), torch.randint(low=0, high=2, size=(1, 1, 32, 32, 32), dtype=torch.int64)) for _ in range(3)]

    EPOCH = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VNet().to(device)
    loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    writer = SummaryWriter()
    
    results, best_model = train_model(
        model, 
        example_train_dataloader, 
        example_test_dataloader,
        loss_fn,
        optimizer,
        accuracy,
        device,
        EPOCH,
        writer
)
    # Open tensorboard
    # %load_ext tensorboard
    # %tensorboard --logdir runs