import os
import torch
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from typing import cast
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

valid_extensions = ('.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.npy', '.npz')

def to_channels (arr: np. ndarray , dtype =np. uint8 ) -> np. ndarray:
    channels = np. unique ( arr)
    res = np. zeros (arr. shape + (len( channels ) ,), dtype = dtype )
    for c in channels :
        c = int(c)
        res [... , c:c +1][ arr == c] = 1
    return res

# load medical image functions. Code obtained from the "Load NIFTI" example code.
def load_data_2D ( imageNames , normImage =False , categorical =False , dtype =np.float32 , getAffines =False , early_stop = False ):
    """
    Load medical image data from names , cases list provided into a list for
    each .
    This function pre - allocates 4D arrays for conv2d to avoid excessive
    memory usage .
    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for
    quick loading and testing scripts .
    """
    affines = []
    # get fixed size
    num = len( imageNames )
    first_case = cast(Nifti1Image, nib.load(imageNames[0])).get_fdata(caching ="unchanged")
    if len ( first_case . shape ) == 3:
        first_case = first_case [:,:,0] # sometimes extra dims , remove
    if categorical :
        first_case = to_channels(first_case)
        rows , cols , channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows , cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = cast(Nifti1Image, nib.load(inName))
        inImage = niftiImage.get_fdata(caching ="unchanged") # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study
        # data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage
            affines.append(affine)
        if i > 20 and early_stop:
            break
    if getAffines:
        return images , affines
    else:
        return images

class HipMRIDataset(torch.utils.data.Dataset):
    def __init__(self, imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False, data_dir=None):
        """Init for the hipMRI dataset class.

        Args:
            imageNames (list of str): List of paths to the NIfTI image files.
            normImage (bool, optional): Whether to normalize the image data. Defaults to False.
            categorical (bool, optional): Whether to convert images to categorical format. Defaults to False.
            dtype (data-type, optional): Desired data type of the loaded images. Defaults to np.float32.
            getAffines (bool, optional): Whether to return the affine matrices along with the images. Defaults to False.
            early_stop (bool, optional): Whether to stop loading after a few images for quick testing. Defaults to False.
            data_dir (str, optional): Directory containing the image files. Defaults to None.
        """
        self.data_dir = data_dir
        self.images = load_data_2D(imageNames, normImage=normImage, categorical=categorical, dtype=dtype, getAffines=getAffines, early_stop=early_stop)
        if getAffines:
            self.images, self.affines = self.images
        else:
            self.affines = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.affines is not None:
            return self.images[idx], self.affines[idx]
        return self.images[idx]

    def normalize_images(self):
        pass

    def search_for_nifti_files(self):
        pass
    
    
    # def __setitem__(self, idx, value):
    #     self.images[idx] = value
    #     if self.affines is not None and isinstance(value, tuple):
    #         self.affines[idx] = value[1]

def normalize(tensor):
    """
    Normalize tensor from [0, 1] to [-1, 1]
    
    Args:
        tensor: Image tensor in range [0, 1]
    Returns:
        Normalized tensor in range [-1, 1]
    """
    return tensor * 2 - 1

# Converting images back too 0,1 range so they may be saved and displayed correctly
def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    
    Args:
        tensor: Image tensor in range [-1, 1]
    Returns:
        Denormalized tensor in range [0, 1]
    """
    return (tensor + 1) / 2

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    imageNames = [
        "test_input/case_040_week_0_slice_0.nii.gz",
        "test_input/case_004_week_0_slice_0.nii.gz"
    ]
    images = load_data_2D(imageNames, getAffines=False, normImage=False, categorical=False, dtype=np.float32, early_stop=False)
    # images = load_data_2D(imageNames, getAffines=False, normImage=True, categorical=False, dtype=np.float32, early_stop=False)
    # print(images.shape)
    print(images[0])

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(images[0], cmap="gray")
    plt.title("MRI")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(images[1], cmap="gray")
    plt.title("MRI 2")
    plt.axis("off")

    plt.show()