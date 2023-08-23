# ⚕️ Segmentation for MR Guided HIFU Uterine Fibroid Ablation

Uterine Fibroids are very common for woman across the globe. Roughly 77% has this health problem. It is a non-cancerous tumour that could grow in or on the uterus. It varies in size and about 1/3 of the cases require surgical removal of the tumour.

An alternative option for non invasive treatment is using MR guided **high-intensity focused ultrasound** surgery (MRgFUS) to perform tissue ablation.

Ultrasound will travel at different speed in the skin, fat, muscle, fibroid, and bone tissue due to different acoustic properties, therefore it is important to run simulations using 3D volumes created by 2D MRI slices.

<p align='center'>
    <img src='https://arrayus.ca/wp-content/uploads/2021/04/Arrayus_treatment_illustration_3_desktop.jpg' width='500'> 
    <h6 align='center'>Arrayus image-guided focused ultrasound platform</h6>
</p>
<br>



# Dataset ℹ️
Consist of 8K image-mask pairs manually segmented. <br>
Due to privacy reasons, this dataset is owned by sunnybrook research institue

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
Also your dataset should be in the same directory as your code. Make sure to change some of the paths in the files as they are based on a computer
```
Parent
| > Code
    | > Seg2D
    ...
| > Dataset
    | > Binary
    ...
```
<br>

There is a patient OOP class in both Seg2D and Seg3D. The Seg2D is more updated and has more features. Both files are called `patient.py` <br>
The sorting of images and masks pairs are also done in `patient.py`

## Data Augmentation
Data Augmentation done in cv2, PIL and torchvision:
- Random Rotation
- Histogram Equalization
- Random Horizontal flip (except for hips)
- Random Colour Jitter
- Normalization

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
<br>

# 2D Slice by Slice Segmentation 🖼️
Individual model per region: Spine, Bowel, Skin, Muscle, Left and Right Hip

## Training or Inferencing
The files that start with the region names are the ones you run. <br>
You can tune hyperparameters in those files, such as the epochs, batch_size, os stuff <br><br>
Example: `spine.py`
```
$ cd Seg2D
$ python3 spine.py
```


When running the file, you can comment the training or inferencing block of code out using `Crtl + /` which looks like this
```
# ================== Training ===================
                        or
# ================== Inference ===================
```

Anything inside the `if __name__ == '__main__': ` is not being run if you import this file as a module to another file. For example the training and inference code won't be run on spine if you import `spine.py` in `skin.py` but the preparation of the spine dataset will be ran.


<br>

For other hyperparmeters like learning rate, type of model, decay rate, optimizer, loss functions etc. 
- If you want to add features into the model go to `models/` directory. 
- If you want to change the hyperparmeters go to `model_run_fn.py` or `model_inference_fn.py`.

<h3 align='center'>For more file information: <a src='https://docs.google.com/document/d/1qI04D95TeWGrsUwWzpSTaq1j02bujT7scYXWUssVvVw/edit?usp=sharing'>Google Docs</a></h3>

<br>

## Model Options
- [Unet](https://arxiv.org/pdf/1505.04597.pdf)
- [Unet++](https://arxiv.org/pdf/1807.10165.pdf)
- [DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf)
- [KMeans](https://en.wikipedia.org/wiki/K-means_clustering)

Haven't debug:
- [Beit3](https://arxiv.org/pdf/2208.10442.pdf)
- [Segformer](https://arxiv.org/pdf/2105.15203.pdf)
- [FPN](https://arxiv.org/pdf/1612.03144.pdf)
- [MANet](https://arxiv.org/pdf/2009.02130.pdf)

More Investigation:
- [SAM](https://arxiv.org/pdf/2304.02643.pdf)

<br>

## Example Output
<p align='center'>
<img src="pictures/spine1.png" width='500'>
<img src="pictures/spine2.png" width='500'>
<img src="pictures/kmean1.png" width='500'>
<img src="pictures/kmeans2.png" width='500'>
</p>

<br> 

# 3D Volumetric Segmentation 🧊
3D Volumetric Segmentation using [Vnet](https://arxiv.org/pdf/1606.04797.pdf) <br>
I have tested and debugged the model but haven't train anything yet

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:2000/1*rcT-PbkROWrSg0PRqO-KAA.png" width=400>
</p>

### How to run:
--- 
Move into the Seg3D directory
```
$ cd Seg3D
$ python3 run.py
```
You can tune the hyperparameters constants in the run.py file

