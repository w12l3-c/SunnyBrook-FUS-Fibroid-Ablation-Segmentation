# ⚕️ Segmentation for MR Guided HIFU Uterine Fibroid Ablation

Uterine Fibroids are very common for woman across the globe. Roughly 77% has this health problem. It is a non-cancerous tumour that could grow in or on the uterus. It varies in size and about 1/3 of the cases require surgical removal of the tumour.

An alternative option for non invasive treatment is using MR guided **high-intensity focused ultrasound** surgery (MRgFUS) to perform tissue ablation.

Ultrasound will travel at different speed in the skin, fat, muscle, fibroid, and bone tissue due to different acoustic properties, therefore it is important to run simulations using 3D volumes created by 2D MRI slices.

<p align='center'>
    <img src='https://arrayus.ca/wp-content/uploads/2021/04/Arrayus_treatment_illustration_3_desktop.jpg'> 
    <h6 align='center'>Arrayus image-guided focused ultrasound platform</h6>
</p>
<br>

# Dataset ℹ️
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

<br>

# Library Dependencies 📚
**To run any code please make a virtual environment first:**
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

# 2D Slice by Slice Segmentation
Each model per region: Spine, Bowel, Skin, Muscle, Left and Right Hip

The files that start with the region names are the ones you run. <br>
etc. spine.py
```
$ cd Seg2D
$ python3 spine.py
```

You can tune hyperparameters in those files, such as the epochs, batch_size, os stuff etc.

For 

## Models
- Unet
- Unet++
- DeepLabv3
- DeepLabv3+
- KMeans

Haven't debug:
- Beit3
- Segformer
- FPN
- MAnet

More Investigation:
- SAM


# 3D Volumetric Segmentation 
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

