from patient import *
from utils import *

import matplotlib.pyplot as plt
import pydicom
import numpy as np

from scipy import ndimage

import torch
import torch.nn.functional as F

from PIL import Image

patient_19_root = os.path.join(mask_root + siemens, listsiemens[4])
print(f"Patient 19 root:")
print(patient_19_root)

patient_19 = Patient(patient_19_root)
patient_19.spine_seg()
patient_19.bowel_seg()
patient_19.hipl_seg()
patient_19.hipr_seg()
patient_19.skin_seg()
patient_19.muscle_seg()

example_dicom = patient_19.spine_img[70]
example_dicom = pydicom.dcmread(example_dicom)

exmaple_dicom2 = patient_19.hipr_img[20]
example_dicom2 = pydicom.dcmread(exmaple_dicom2)

print(f"Example dicom")
print(example_dicom.ImageOrientationPatient)    # Unit vector for P and S
print(example_dicom.ImagePositionPatient)   # X Y Z of L

spine_stack, spine_mask_stack = stack3d(patient_19.spine_img, patient_19.spine_mask)

# volume = (channel, depth, height, width)
print(spine_stack.shape, spine_mask_stack.shape)

# Batch size = 1
spine_stack = np.expand_dims(spine_stack, axis=0)
spine_mask_stack = np.expand_dims(spine_mask_stack, axis=0)
print(spine_stack.shape, spine_mask_stack.shape)

# Show example slice
example_mask = cv2.imread(patient_19.spine_mask[70])

# Show mask on matplotlib
fig, ax = plt.subplots(1, 3, figsize=(10, 10))
ax[0].imshow(example_dicom.pixel_array, cmap='gray')
ax[0].set_title(f'Size of dicom: {example_dicom.pixel_array.shape}')
ax[1].imshow(example_mask, cmap='gray')
ax[1].set_title(f'Size of mask: {example_mask.shape}')
ax[2].imshow(example_dicom.pixel_array, cmap='gray')
ax[2].imshow(example_mask, cmap='jet', alpha=0.5)
ax[2].set_title('Overlay dicom and mask')
plt.show()
plt.clf()

# Shrinking the image
spine_shrink = F.interpolate(torch.from_numpy((spine_stack).astype(np.float32)), size=(128, 128, 128), mode='trilinear', align_corners=False)
print(spine_shrink.shape)

example_shrink = spine_shrink[0, 0, 60, :, :]
example_shrink = example_shrink.numpy()
example_shrink = np.squeeze(np.squeeze(example_shrink))

example_shrink = Image.fromarray(example_shrink)
print(example_shrink.size)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(example_dicom.pixel_array, cmap='gray')
ax[0].set_title(f'Before Interpolation')
ax[1].imshow(example_shrink, cmap='gray')
ax[1].set_title(f'After Interpolation')
plt.show()