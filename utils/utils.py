import random
import numpy as np
import SimpleITK as sitk
import os

def readimage(filename):
    return sitk.ReadImage(filename)

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

def resize_and_crop(pilimg, scale=0.5, final_height=None):

    n = sitk.GetArrayFromImage(pilimg)
    new_n = n[1:49, 1:49]
    
    return new_n

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def channalize_mask(mask):
    
    new_mask = np.zeros((48, 48, 6), dtype = np.float)
    
    for i in range(0, 48):
        
        for j in range(0, 48):
            
            if mask[i, j] != 0:

                new_mask[i, j, mask[i, j] - 1] = 1
    
    return new_mask


