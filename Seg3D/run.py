import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from dataclass import *
from patient import *
from utils import *
from vnet import *
from train import *
from Seg3D.save_load_3D import *
from eval import *


# ---------------------- Hyperparameters ---------------------- #
EPOCHS = 100
BATCH_SIZE = 1
LR = 1e-4
CLASSES = sorted(['Spine', 'hip_R', 'hip_L', 'Skin', 'Muscle', 'Bowel'])
NUM_WORKERS = 4     # Maybe 2 or 8
MRI_SIZE = (128, 128, 128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------- Main Roots ---------------------- #
mask_root = "/mnt/HDD_1TB/Wallace/Segmentation_Raw/"
subdir1 = sorted(os.listdir(mask_root))

dicom_root = "/mnt/HDD_1TB/Wallace/Datasets/fibroid trial HIFU-SB-001/"
new_dicom_root = "/mnt/HDD_1TB/Wallace/Datasets/fibroid trial HIFU-SB-001/Siemens/"
old_dicom_root = "/mnt/HDD_1TB/Wallace/Datasets/fibroid trial HIFU-SB-001/Arrayus/"

bones = subdir1[0]
listbones = sorted(os.listdir(mask_root + bones))
siemens = subdir1[1]
listsiemens = sorted(os.listdir(mask_root + siemens))
lowres = subdir1[2]
listlowres = sorted(os.listdir(mask_root + lowres))
arrayus = subdir1[3]
listarrayus = sorted(os.listdir(mask_root + arrayus))


# ---------------------- Data ---------------------- #
img_list = create_img_list()
mask_list = create_mask_list()

train_img_list, train_mask_list, val_img_list, val_mask_list = train_test_split(img_list, mask_list, 0.8)

train_dataloader = create_dataloader(train_img_list, train_mask_list, BATCH_SIZE, True, NUM_WORKERS, True)
val_dataloader = create_dataloader(val_img_list, val_mask_list, BATCH_SIZE, False, NUM_WORKERS, True)


# ---------------------- Model ---------------------- #
VNET = VNet().to(device)
loss_fn = DiceLoss()
optimizer = torch.optim.AdamW(VNET.parameters(), lr=LR)
writer = SummaryWriter()

result = train_model(
    VNET,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    accuracy,
    device,
    EPOCHS,
    writer
)



