import streamlit as st
import torch
import torchvision

import os
import cv2
import pydicom
import dicom2nifti
import numpy as np
from PIL import Image

dicom_folder = st.file_uploader("Upload a DICOM Folder", type=["dir", "dcm"], accept_multiple_files=True)
dicom_folder.sort(key=lambda x: x.name)

st.write("--------------------------------------------------")

mask_button = st.checkbox("Mask Avaliable?")
if mask_button:
    mask_folder = st.file_uploader("Upload a Mask Folder", type=["dir", "png"], accept_multiple_files=True)
    mask_folder.sort(key=lambda x: x.name)
    st.write("--------------------------------------------------")
    
if dicom_folder:
    slice = st.slider("DICOM File Slice", 0, len(dicom_folder)-1, 50)
    try:
        patient_slice = dicom_folder[slice]
        patient_slice = pydicom.dcmread(patient_slice)
        
        st.write("--------------------------------------------------")
        st.write(f"{patient_slice.PatientName}: {slice}")
        
        img = patient_slice.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        st.image(img, caption="DICOM Image", use_column_width=True)
    
        trained_models = sorted(os.listdir('/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models'))
        
    except Exception as e:
        st.error("Error reading DICOM file: " + str(e))