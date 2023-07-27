import torch
import torch.nn as nn

import tensorflow as tf
import tensorflow.keras as keras

# ---------------------- Pytorch ---------------------- #
def save_model_torch(model, path):
    torch.save(model.state_dict(), path)
    
def load_model_torch(model, path):
    model.load_state_dict(torch.load(path))
    return model


# ---------------------- Tensorflow ---------------------- #
def save_model_tf(model, path):
    model.save(path)

def load_model_tf(path):
    model = keras.models.load_model(path)
    return model