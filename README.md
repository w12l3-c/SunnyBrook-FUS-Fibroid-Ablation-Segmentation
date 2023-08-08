# ⚕️ Segmentation for MR Guided HIFU Uterine Fibroid Ablation

Uterine Fibroid is a very common disease for woman across the globe. It is a non-cancerous tumour that could grow in or on the uterus. It varies in size and about 1/3 of the cases require surgical removal of the tumour.

An alternative option for non invasive treatment is using MR guided **high-intensity focused ultrasound** (HIFU) to perform tissue ablation.

<p align='center'>
    <img src='https://corporate.webassets.siemens-healthineers.com/1800000005950114/584d3b946cd1/v/48bf69ac4411/Exablate-Neuro-with-Siemens-Skyra-High-res-dark_portlet_1800000005950114.jpg'>
</p>

In order to focus the ultrasound on the focal point, the transducers have to be phased according to the time it travel throught the body medium. 

Ultrasound will travel at different speed in the skin, fat, muscle, fibroid, and bone tissue, therefore it is important to run simulations using 3D volumes created by 2D MRI slices.

This repository is the proof of concept to show that the deep learning is a good approach for segmenting these regions in the MRI slices.

## Dataset ℹ️
Consist of 8K image-mask pairs manually segmented.

Regions include:
- Spine
- Bowel
- Muscle
- Skin
- Hip RL

To prepare your own dataset, it should be in the format of:
```
Dataset
| > Binary
    | >  Spine
    | >  Bowel
    | >  etc.
| > Img
    | >  Coronal
    | >  Sagittal
| > VOI
    | >  Spine
    | >  Bowel
    | >  etc.
```

## Library Dependencies 📚
To run any code please make a virtual environment first:
```
$ python3 -m venv DIR
```
To activate the virtual environment:
```
$ source VENV_DIR/bin/activate
```
You should see the terminal has:
```
$ (.venv)(base)
```
Then install the library dependencies:
```
$ pip3 install -r requirements.txt
``` 
To deactivate the virtural environment:
```
$ deactivate
```

## 2D Slice by Slice Segmentation
Each model per region:



## 3D Volumetric Segmentation 
3D Volumetric Segmentation using [Vnet](https://arxiv.org/pdf/1606.04797.pdf) 

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:2000/1*rcT-PbkROWrSg0PRqO-KAA.png" width=400>
</p>

### How to run:
--- 
Move into the Seg3D directory
```
$ cd Seg3D
```
You can tune the hyperparameters constants in the run.py file
```
$ python3 run.py
```

