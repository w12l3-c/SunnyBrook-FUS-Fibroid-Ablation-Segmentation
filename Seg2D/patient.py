# =============================================================================
# File Description:
# =====================
# This file contains the class Patient class where
# the patient's data is processed into images and masks
# in a format that is readable by other functions such as dataloader
# =============================================================================

# ====================== Imports ====================== #
import os
import re

# ====================== Main Roots ====================== #
# Main Root for Masks
mask_root = "/mnt/HDD_1TB/Wallace/Segmentation_Raw_2D/"
# Sub-Directories of Main
subdir1 = sorted(os.listdir(mask_root))

# Siemens dataset with mask only for bones (spine, hips)
bones = subdir1[0]
# List of all patient paths in this directory
listbones = sorted(os.listdir(mask_root + bones))
listbones = [os.path.join(mask_root + bones, i) for i in listbones]

# Siemens dataset with mask for every class
siemens = subdir1[1]
# List of all patient paths in this directory
listsiemens = sorted(os.listdir(mask_root + siemens))
listsiemens= [os.path.join(mask_root + siemens, i) for i in listsiemens]

# Siemens dataset with low resolution mri
lowres = subdir1[2]
# List of all patient paths in this directory
listlowres = sorted(os.listdir(mask_root + lowres))
listlowres = [os.path.join(mask_root + lowres, i) for i in listlowres]

# GE Arrayus dataset with mask for every class
arrayus = subdir1[3]
# List of all patient paths in this directory
listarrayus = sorted(os.listdir(mask_root + arrayus))
listarrayus = [os.path.join(mask_root + arrayus, i) for i in listarrayus]

# Image Size of the dicom images for each class
siemens_size = (320, 320)
bones_size = (160, 320)
arrayus_size = (256, 256)

# ====================== Patient Class ====================== #
class Patient:
    def __init__(self, path) -> None:
        # Patient root path
        self.path = path    
        self.dir = sorted(os.listdir(path))
        
        # Patient number
        self.name = (path.split('/')[-1]).split('-')[0]
        
        # Mask and Dicom(Img) directory
        self.mask_dir = os.path.join(path, self.dir[0])
        self.dicom_dir = os.path.join(path, self.dir[1])
        
        self.mask_listdir = sorted(os.listdir(self.mask_dir))   # ['Bowel', 'Catheter', 'Muscle', 'Skin', 'Spine', 'hip_L', 'hip_R']
        self.dicom_listdir = sorted(os.listdir(self.dicom_dir))
        
        # Dicom(Img) directory for coronal and sagittal
        self.coronal = os.path.join(self.dicom_dir, self.dicom_listdir[0])
        self.sagittal = os.path.join(self.dicom_dir, self.dicom_listdir[1])
        
        self.coronal_listdir = sorted(os.listdir(self.coronal))
        self.sagittal_listdir = sorted(os.listdir(self.sagittal))
        
        # Colour palette for multi-class segmentation
        self.colour_label = {'Spine': (35, 132, 250), 'Bowel': (14, 240, 56), 'Muscle': (214, 51, 36), 'Skin': (240, 170, 31), 'hip_L': (173, 20, 250), 'hip_R': (131, 20, 250)}
        self.colour_list = [(35, 132, 250), (14, 240, 56), (214, 51, 36), (240, 170, 31), (173, 20, 250), (131, 20, 250)]
        
        # Sagittal and coronal spacing
        self.sagittal_spacing = [0.9375, 0.9375]
        
        # LPS coordinates
        sagittal_slices = len(self.sagittal_listdir)
        coronal_slices = len(self.coronal_listdir)
        self.LPS = [sagittal_slices, coronal_slices, coronal_slices]
        
    def spine_seg(self):
        # Spine masks directory
        self.spine_dir = os.path.join(self.mask_dir, 'Spine')
        self.spine_listdir = sorted(os.listdir(self.spine_dir))
        
        # Start and end index of spine masks
        start = int(re.findall(r'\d+', self.spine_listdir[0])[0])
        end = int(re.findall(r'\d+', self.spine_listdir[-1])[0])
        
        # Spine masks and corresponding images' paths in lists
        self.spine_mask = [os.path.join(self.spine_dir, x) for x in self.spine_listdir]
        self.spine_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir[start-1:end]]
        
    def bowel_seg(self):
        # Bowel masks directory
        self.bowel_dir = os.path.join(self.mask_dir, 'Bowel')
        self.bowel_listdir = sorted(os.listdir(self.bowel_dir))
        
        # Start and end index of bowel masks
        start = int(re.findall(r'\d+', self.bowel_listdir[0])[0])
        end = int(re.findall(r'\d+', self.bowel_listdir[-1])[0])
        
        # Bowel masks and corresponding images' paths in lists
        self.bowel_mask = [os.path.join(self.bowel_dir, x) for x in self.bowel_listdir]
        self.bowel_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir[start-1:end]]
        
    def hipl_seg(self):
        # Left Hip masks directory
        self.hipl_dir = os.path.join(self.mask_dir, 'hip_L')
        self.hipl_listdir = sorted(os.listdir(self.hipl_dir))
        
        # Start and end index of hipl masks
        start = int(re.findall(r'\d+', self.hipl_listdir[0])[0])
        end = int(re.findall(r'\d+', self.hipl_listdir[-1])[0])
        
        # Left Hip masks and corresponding images' paths in lists
        self.hipl_mask = [os.path.join(self.hipl_dir, x) for x in self.hipl_listdir]
        self.hipl_img = [os.path.join(self.coronal, x) for x in self.coronal_listdir[start-1:end]]
        
    def hipr_seg(self):
        # Right Hip masks directory
        self.hipr_dir = os.path.join(self.mask_dir, 'hip_R')
        self.hipr_listdir = sorted(os.listdir(self.hipr_dir))
        
        # Start and end index of hipr masks
        start = int(re.findall(r'\d+', self.hipr_listdir[0])[0])
        end = int(re.findall(r'\d+', self.hipr_listdir[-1])[0])
        
        # Right Hip masks and corresponding images' paths in lists
        self.hipr_mask = [os.path.join(self.hipr_dir, x) for x in self.hipr_listdir]
        self.hipr_img = [os.path.join(self.coronal, x) for x in self.coronal_listdir[start-1:end]]
        
    def skin_seg(self):
        # Skin Fat masks directory
        self.skin_dir = os.path.join(self.mask_dir, 'Skin')
        self.skin_listdir = sorted(os.listdir(self.skin_dir))
        
        # Start and end index of skin masks
        start = int(re.findall(r'\d+', self.skin_listdir[0])[0])
        end = int(re.findall(r'\d+', self.skin_listdir[-1])[0])
        
        # Skin Fat masks and corresponding images' paths in lists
        self.skin_mask = [os.path.join(self.skin_dir, x) for x in self.skin_listdir]
        self.skin_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir[start-1:end]]
        
    def muscle_seg(self):
        # Muscle masks directory
        self.muscle_dir = os.path.join(self.mask_dir, 'Muscle')
        self.muscle_listdir = sorted(os.listdir(self.muscle_dir))
        
        # Start and end index of muscle masks
        start = int(re.findall(r'\d+', self.muscle_listdir[0])[0])
        end = int(re.findall(r'\d+', self.muscle_listdir[-1])[0])
        
        # Muscle masks and corresponding images' paths in lists
        self.muscle_mask = [os.path.join(self.muscle_dir, x) for x in self.muscle_listdir]
        self.muscle_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir[start-1:end]]

    def inference_seg(self):
        # Sagittal Dataset
        self.sagittal_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir]

        # Coronal Dataset
        self.coronal_img = [os.path.join(self.coronal, x) for x in self.coronal_listdir]

    def multilabel_seg(self):
        # Multilabel masks directory
        smallest_index = 10000  # Initialize with a large number
        largest_index = 0    # Initialize with a small number
        
        # Sagittal and Coronal parts
        sag_list = ['Bowel', 'Spine', 'Skin', 'Muscle']
        cor_list = ['hip_L', 'hip_R']
        
        # Sagittal
        # Loop through the regions
        for region in sag_list:
            # Region directory
            region_dir = os.path.join(self.mask_dir, region)
            if os.path.exists(region_dir):
                # List of masks in the region directory
                region_listdir = sorted(os.listdir(region_dir))

                # Find the start and end of the region
                start = int(re.findall(r'\d+', region_listdir[0])[0])
                end = int(re.findall(r'\d+', region_listdir[-1])[0])
                
                # Update the smallest and largest index
                # This is to find the first slice and last slice of the patient that has at least one region
                if start < smallest_index:
                    smallest_index = start
                
                if end > largest_index:
                    largest_index = end
        
        # Create list for mask and image paths
        self.multilabel_sag_mask = []
        self.multilabel_sag_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir[smallest_index-1:largest_index]]
        
        # Loop through the slices
        for i in range(smallest_index, largest_index+1):
            mask_stack = [] # Each slice might have more than 1 mask
            # Loop through each region
            for region in sag_list:
                region_dir = os.path.join(self.mask_dir, region)
                if os.path.exists(region_dir):
                    # Create list from the region directory
                    region_listdir = sorted(os.listdir(region_dir))
                    # Loop through the directory to find the mask that has the same slice number
                    for path in region_listdir:
                        if int(re.findall(r'\d+', path)[0]) == i:
                            mask_stack.append(os.path.join(region_dir, path))
            # Append 1 mask stack to the entire mask dataset
            self.multilabel_sag_mask.append(mask_stack)
        
        # Coronal 
        smallest_index = 10000  # Initialize with a large number
        largest_index = 0   # Initialize with a small number
        
        # Loop through the regions
        for region in cor_list:
            # Region directory
            region_dir = os.path.join(self.mask_dir, region)
            if os.path.exists(region_dir):
                # List of masks in the region directory
                region_listdir = sorted(os.listdir(region_dir))
            
                start = int(re.findall(r'\d+', region_listdir[0])[0])
                end = int(re.findall(r'\d+', region_listdir[-1])[0])
                
                # Find the largest and smallest index in the patient
                if start < smallest_index:
                    smallest_index = start
                
                if end > largest_index:
                    largest_index = end
        
        # Create list for mask and image paths
        self.multilabel_cor_mask = []
        self.multilabel_cor_img = [os.path.join(self.coronal, x) for x in self.coronal_listdir[smallest_index-1:largest_index]]
        
        # Loop through the slices
        for i in range(smallest_index, largest_index+1):
            mask_stack = [] # Each slice might have more than 1 mask
            # Loop through each region
            for region in cor_list:
                region_dir = os.path.join(self.mask_dir, region)
                if os.path.exists(region_dir):
                    # Create list from the region directory
                    region_listdir = sorted(os.listdir(region_dir))
                    for path in region_listdir:
                        # Loop through the directory to find the mask that has the same slice number
                        if int(re.findall(r'\d+', path)[0]) == i:
                            mask_stack.append(os.path.join(region_dir, path))
            # Append 1 mask stack to the entire mask dataset
            self.multilabel_cor_mask.append(mask_stack)
        
        # print(f"Multilabel Sag Mask: {self.multilabel_sag_mask[50:100]}")
        # print(f"Multilabel Cor Mask: {self.multilabel_cor_mask[:3]}")
        
    def prepare(self):
        # Run all segmentation file paths functions above base on the avaliable folder in patient
        # ie. Spine masks is available in all patients, but Muscle is not
        if 'Spine' in self.mask_listdir:
            self.spine_seg()
        if 'Bowel' in self.mask_listdir:
            self.bowel_seg()
        if 'hip_L' in self.mask_listdir:
            self.hipl_seg()
        if 'hip_R' in self.mask_listdir:
            self.hipr_seg()
        if 'Skin' in self.mask_listdir:
            self.skin_seg()
        if 'Muscle' in self.mask_listdir:
            self.muscle_seg()
        
        self.inference_seg()
        self.multilabel_seg()
            
            
# ===================== Create Patient List ===================== #
def create_patient_list(path_list=listsiemens):
    # Full list of patient images
    patient_list = []
    
    # Loop over every patient 
    for path in path_list:
        # Instantiate Patient class for every patient and prepare the segmentation paths
        patient = Patient(path)
        patient.prepare()
        
        # Append each patient to the list
        patient_list.append(patient)
    
    # Return a list of patient class object
    return patient_list
    
    
# ============= Convert Patient List to Torch Dataset Style ============ #    
def get_spine(patient_list):
    """
    Extract spine images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing spine images and masks.
    """
    # Create an empty list to store image and mask pairs.
    path_list = []

    # Loop through each patient in the patient_list.
    for patient in patient_list:
        # Check if the patient object has 'spine_img' attribute.
        if hasattr(patient, 'spine_img'):
            # Iterate over the spine images of the patient.
            for index, img in enumerate(patient.spine_img):
                # Retrieve the corresponding mask for the current spine image.
                mask = patient.spine_mask[index]
                # Create a dictionary containing the image and mask and add it to the path_list.
                path_list.append({'img': img, 'mask': mask})

    return path_list
    
def get_bowel(patient_list):
    """
    Extract bowel images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing bowel images and masks.
    """
    # Create an empty list to store image and mask pairs.
    path_list = []
    # Loop through each patient in the patient_list.
    for patient in patient_list:
        # Check if the patient object has 'bowel_img' attribute.
        if hasattr(patient, 'bowel_img'):
            # Iterate over the bowel images of the patient.
            for index, img in enumerate(patient.bowel_img):
                # Retrieve the corresponding mask for the current bowel image.
                mask = patient.bowel_mask[index]
                # Create a dictionary containing the image and mask and add it to the path_list.
                path_list.append({'img':img, 'mask':mask})

    return path_list

def get_hipl(patient_list):
    """
    Extract left hip images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing left hip images and masks.
    """
    # Create an empty list to store image and mask pairs.
    path_list = []
    # Loop through each patient in the patient_list.
    for patient in patient_list:
        # Check if the patient object has 'hipl_img' attribute.
        if hasattr(patient, 'hipl_img'):
            # Iterate over the left hip images of the patient.
            for index, img in enumerate(patient.hipl_img):
                # Retrieve the corresponding mask for the current left hip image.
                mask = patient.hipl_mask[index]
                # Create a dictionary containing the image and mask and add it to the path_list.
                path_list.append({'img':img, 'mask':mask})
    
    return path_list

def get_hipr(patient_list):
    """
    Extract right hip images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing right hip images and masks.
    """
    # Create an empty list to store image and mask pairs.
    path_list = []
    # Loop through each patient in the patient_list.
    for patient in patient_list:
        # Check if the patient object has 'hipr_img' attribute.
        if hasattr(patient, 'hipr_img'):
            # Iterate over the right hip images of the patient.
            for index, img in enumerate(patient.hipr_img):
                # Retrieve the corresponding mask for the current right hip image.
                mask = patient.hipr_mask[index]
                # Create a dictionary containing the image and mask and add it to the path_list.
                path_list.append({'img':img, 'mask':mask})
    
    return path_list

def get_muscle(patient_list):
    """
    Extract abdominal muscle images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing abdominal muscle images and masks.
    """
    # Create an empty list to store image and mask pairs.
    path_list = []
    # Loop through each patient in the patient_list.
    for patient in patient_list:
        # Check if the patient object has 'muscle_img' attribute.
        if hasattr(patient, 'muscle_img'):
            # Iterate over the abdominal muscle images of the patient.
            for index, img in enumerate(patient.muscle_img):
                # Retrieve the corresponding mask for the current abdominal muscle image.
                mask = patient.muscle_mask[index]
                # Create a dictionary containing the image and mask and add it to the path_list.
                path_list.append({'img':img, 'mask':mask})
    return path_list

def get_skin(patient_list):
    """
    Extract skin and fat images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing skin and fat images and masks.
    """
    path_list = []
    for patient in patient_list:
        if hasattr(patient, 'skin_img'):
            for index, img in enumerate(patient.skin_img):
                mask = patient.skin_mask[index]
                path_list.append({'img':img, 'mask':mask})
        
    return path_list

def get_inference(patient_list):
    """
    Extract sagittal and coronal images without masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        tuple: A tuple of two lists containing sagittal and coronal images without masks.
    """
    path_list_sagittal = []
    path_list_coronal = []
    for patient in patient_list:
        for index, img in enumerate(patient.sagittal_img):
            mask = None
            path_list_sagittal.append({'img':img, 'mask':mask})
    for patient in patient_list:
        for index, img in enumerate(patient.coronal_img):
            mask = None
            path_list_coronal.append({'img':img, 'mask':mask})
            
    return path_list_sagittal, path_list_coronal
    
def get_multilabel_coronal(patient_list):
    """
    Extract multi-label coronal images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing multi-label coronal images and masks.
    """
    path_list = []
    for patient in patient_list:
        for index, img in enumerate(patient.multilabel_cor_img):
            mask = patient.multilabel_cor_mask[index]
            path_list.append({'img':img, 'mask':mask})
    return path_list

def get_multilabel_sagittal(patient_list):
    """
    Extract multi-label sagittal images and masks from a list of patients.

    Args:
        patient_list (list): A list of patient objects.

    Returns:
        list: A list of dictionaries containing multi-label sagittal images and masks.
    """
    path_list = []
    for patient in patient_list:
        for index, img in enumerate(patient.multilabel_sag_img):
            mask = patient.multilabel_sag_mask[index]
            path_list.append({'img':img, 'mask':mask})
    return path_list

# ===================== Test ===================== #
if __name__ == '__main__':
    patient_19_root = os.path.join(mask_root + lowres, listlowres[0])
    patient_19 = Patient(patient_19_root)
    
    

