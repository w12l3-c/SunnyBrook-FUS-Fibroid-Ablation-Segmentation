# =============================================================================
# File Description:
# ------------------
# Pytorch:
# This file is contains the function to save the model.state_dict as a pth file
# and load the model.state_dict from a pth file 
#
# Tensorflow:
# Save and load tensorlfow model
# =============================================================================

# =================== Imports =================== #
import torch
import torch.nn as nn

# import tensorflow as tf
# import tensorflow.keras as keras

# ========================= Pytorch ========================= #
def save_model_torch(model_state_dict, path):
    """
    Save the architecture and params of a PyTorch model to a pth file.

    Args:
        model_state_dict (dict): The state dictionary of the PyTorch model.
        path (str): The file path where the model state dictionary will be saved.
    """
    torch.save(model_state_dict, path)  # Save Model
    
def load_model_torch(model, path):
    """
    Load a PyTorch model with its state dictionary from a pth file.

    Args:
        model (torch.nn.Module): An instance of a PyTorch model where the state dictionary will be loaded.
        path (str): The file path from which the model state dictionary will be loaded.

    Returns:
        torch.nn.Module: The PyTorch model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(path), strict=False)   # Load Model
    return model

# ======================== Tensorflow ======================== #
# def save_model_tf(model, path):
#     model.save(path)

# def load_model_tf(path):
#     model = keras.models.load_model(path)
#     return model