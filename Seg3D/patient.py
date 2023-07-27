import os
import re

# ---------------------- Main Roots ---------------------- #
mask_root = "/mnt/HDD_1TB/Wallace/Segmentation_Raw/"
subdir1 = sorted(os.listdir(mask_root))

dicom_root = "/mnt/HDD_1TB/Wallace/Datasets/fibroid trial HIFU-SB-001/"
new_dicom_root = "/mnt/HDD_1TB/Wallace/Datasets/fibroid trial HIFU-SB-001/Siemens/"
old_dicom_root = "/mnt/HDD_1TB/Wallace/Datasets/fibroid trial HIFU-SB-001/Arrayus/"

bones = subdir1[0]
listbones = sorted(os.listdir(mask_root + bones))
listbones = [os.path.join(mask_root + bones, i) for i in listbones]

siemens = subdir1[1]
listsiemens = sorted(os.listdir(mask_root + siemens))
listsiemens= [os.path.join(mask_root + siemens, i) for i in listsiemens]

lowres = subdir1[2]
listlowres = sorted(os.listdir(mask_root + lowres))
listlowres = [os.path.join(mask_root + lowres, i) for i in listlowres]

arrayus = subdir1[3]
listarrayus = sorted(os.listdir(mask_root + arrayus))
listarrayus = [os.path.join(mask_root + arrayus, i) for i in listarrayus]


# ---------------------- Additional Note ---------------------- #
# If this is actually used as part of clinical practice
# they should manualy tune the brightness and contrast on the dicom and save it 


# ---------------------- Patient Class ---------------------- #
class Patient:
    def __init__(self, path) -> None:
        self.path = path
        self.dir = sorted(os.listdir(path))
        self.mask_dir = os.path.join(path, self.dir[0])
        self.dicom_dir = os.path.join(path, self.dir[1])
        
        self.mask_listdir = sorted(os.listdir(self.mask_dir))   # ['Bowel', 'Catheter', 'Muscle', 'Skin', 'Spine', 'hip_L', 'hip_R']
        self.dicom_listdir = sorted(os.listdir(self.dicom_dir))
        
        self.coronal = os.path.join(self.dicom_dir, self.dicom_listdir[0])
        self.sagittal = os.path.join(self.dicom_dir, self.dicom_listdir[1])
        
        self.coronal_listdir = sorted(os.listdir(self.coronal))
        self.sagittal_listdir = sorted(os.listdir(self.sagittal))
        
        self.colour_label = {'Spine': (35, 132, 250), 'Bowel': (14, 240, 56), 'Muscle': (214, 51, 36), 'Skin': (240, 170, 31), 'hip_L': (173, 20, 250), 'hip_R': (131, 20, 250)}
        
        self.sagittal_spacing = [0.9375, 0.9375]
        
        self.name = (path.split('/')[-1]).split('-')[0]
        
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
        self.spine_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir]
        
    def bowel_seg(self):
        # Bowel masks directory
        self.bowel_dir = os.path.join(self.mask_dir, 'Bowel')
        self.bowel_listdir = sorted(os.listdir(self.bowel_dir))
        
        # Start and end index of bowel masks
        start = int(re.findall(r'\d+', self.bowel_listdir[0])[0])
        end = int(re.findall(r'\d+', self.bowel_listdir[-1])[0])
        
        # Bowel masks and corresponding images' paths in lists
        self.bowel_mask = [os.path.join(self.bowel_dir, x) for x in self.bowel_listdir]
        self.bowel_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir]
        
    def hipl_seg(self):
        # Left Hip masks directory
        self.hipl_dir = os.path.join(self.mask_dir, 'hip_L')
        self.hipl_listdir = sorted(os.listdir(self.hipl_dir))
        
        # Start and end index of hipl masks
        start = int(re.findall(r'\d+', self.hipl_listdir[0])[0])
        end = int(re.findall(r'\d+', self.hipl_listdir[-1])[0])
        
        # Left Hip masks and corresponding images' paths in lists
        self.hipl_mask = [os.path.join(self.hipl_dir, x) for x in self.hipl_listdir]
        self.hipl_img = [os.path.join(self.coronal, x) for x in self.coronal_listdir]
        
    def hipr_seg(self):
        # Right Hip masks directory
        self.hipr_dir = os.path.join(self.mask_dir, 'hip_R')
        self.hipr_listdir = sorted(os.listdir(self.hipr_dir))
        
        # Start and end index of hipr masks
        start = int(re.findall(r'\d+', self.hipr_listdir[0])[0])
        end = int(re.findall(r'\d+', self.hipr_listdir[-1])[0])
        
        # Right Hip masks and corresponding images' paths in lists
        self.hipr_mask = [os.path.join(self.hipr_dir, x) for x in self.hipr_listdir]
        self.hipr_img = [os.path.join(self.coronal, x) for x in self.coronal_listdir]
        
    def skin_seg(self):
        # Skin Fat masks directory
        self.skin_dir = os.path.join(self.mask_dir, 'Skin')
        self.skin_listdir = sorted(os.listdir(self.skin_dir))
        
        # Start and end index of skin masks
        start = int(re.findall(r'\d+', self.skin_listdir[0])[0])
        end = int(re.findall(r'\d+', self.skin_listdir[-1])[0])
        
        # Skin Fat masks and corresponding images' paths in lists
        self.skin_mask = [os.path.join(self.skin_dir, x) for x in self.skin_listdir]
        self.skin_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir]
        
    def muscle_seg(self):
        # Muscle masks directory
        self.muscle_dir = os.path.join(self.mask_dir, 'Muscle')
        self.muscle_listdir = sorted(os.listdir(self.muscle_dir))
        
        # Start and end index of muscle masks
        start = int(re.findall(r'\d+', self.spine_listdir[0])[0])
        end = int(re.findall(r'\d+', self.spine_listdir[-1])[0])
        
        # Muscle masks and corresponding images' paths in lists
        self.muscle_mask = [os.path.join(self.muscle_dir, x) for x in self.muscle_listdir]
        self.muscle_img = [os.path.join(self.sagittal, x) for x in self.sagittal_listdir]

    def prepare(self):
        if 'Spine' in self.mask_listdir:
            self.spine_seg()
        elif 'Bowel' in self.mask_listdir:
            self.bowel_seg()
        elif 'hip_L' in self.mask_listdir:
            self.hipl_seg()
        elif 'hip_R' in self.mask_listdir:
            self.hipr_seg()
        elif 'Skin' in self.mask_listdir:
            self.skin_seg()
        elif 'Muscle' in self.mask_listdir:
            self.muscle_seg()
        
        
def create_patient_list():
    # Full list of patient images
    patient_list = []
    path_list = listbones + listarrayus + listlowres + listsiemens
    for path in path_list:
        patient = Patient(path)
        patient.prepare()
        element = {
            patient.name: {
                'img':{
                    'spine':patient.spine_img if hasattr(patient, 'spine_img') else None,
                    'bowel':patient.bowel_img if hasattr(patient, 'bowel_img') else None,
                    'hip_l':patient.hipl_img if hasattr(patient, 'hipl_img') else None,
                    'hip_r':patient.hipr_img if hasattr(patient, 'hipr_img') else None,
                    'skin':patient.skin_img if hasattr(patient, 'skin_img') else None,
                    'muscle':patient.muscle_img if hasattr(patient, 'muscle_img') else None
                },
                'mask':{
                    'spine':patient.spine_mask if hasattr(patient, 'spine_mask') else None,
                    'bowel':patient.bowel_mask if hasattr(patient, 'bowel_mask') else None,
                    'hip_l':patient.hipl_mask if hasattr(patient, 'hipl_mask') else None,
                    'hip_r':patient.hipr_mask if hasattr(patient, 'hipr_mask') else None,
                    'skin':patient.skin_mask if hasattr(patient, 'skin_mask') else None,
                    'muscle':patient.muscle_mask if hasattr(patient, 'muscle_mask') else None
                }
            }
        }
        patient_list.append(element)
    return patient_list
    
    
# [
#     self.name: {
#         'img':{
#             'spine':self.spinedir,
#             'bowel':self.boweldir,
#             'hip_l':self.hipldir,
#             'hip_r':self.hiprdir,
#             'skin':self.skindir,
#             'muscle':self.muscledir   
#         },
#         'mask':{
#             'spine':self.spinedir,
#             'bowel':self.boweldir,
#             'hip_l':self.hipldir,
#             'hip_r':self.hiprdir,
#             'skin':self.skindir,
#             'muscle':self.muscledir 
#         }
#     }
# ]