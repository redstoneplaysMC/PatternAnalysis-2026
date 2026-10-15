from random import Random, seed

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import cast
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from pathlib import Path
import re
import yaml

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

class HipMRIDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_images=None,
        norm_image=True,
        dtype=np.float32,
        week=None,
        cases=None,
    ):
        self.data_dir = Path(data_dir)
        self.norm_image = norm_image
        self.dtype = dtype

        self.image_paths = sorted(
            p for p in self.data_dir.rglob("*.nii.gz")
            if p.is_file()
            and re.search(r"case_\d+_week_4_slice_\d+", p.name)
        )

        # Filter by week and/or case using filenames.
        filtered_paths = []

        for path in self.image_paths:
            match = re.search(
                r"case_(\d+)_week_(\d+)_slice_(\d+)",
                path.name,
            )

            if match is None:
                continue

            case_id = int(match.group(1))
            week_id = int(match.group(2))

            if week is not None and week_id != week:
                continue

            if cases is not None and case_id not in cases:
                continue

            filtered_paths.append(path)

        self.image_paths = filtered_paths

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if not self.image_paths:
            raise ValueError(
                "No matching NIfTI files found. "
                "Check data_dir, week and cases."
            )

        self.slice_indices = []

        for file_idx, path in enumerate(self.image_paths):
            img = nib.load(str(path))

            if len(img.shape) == 2: #type:ignore
                num_slices = 1
            elif len(img.shape) == 3: #type:ignore
                num_slices = img.shape[2] #type:ignore
            else:
                raise ValueError(
                    f"Unexpected NIfTI shape {img.shape}: {path}" #type:ignore
                )

            self.slice_indices.extend(
                (file_idx, slice_idx)
                for slice_idx in range(num_slices)
            )

        print(f"NIfTI files found: {len(self.image_paths)}")
        print(f"2D slices indexed: {len(self.slice_indices)}")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_indices[idx]
        path = self.image_paths[file_idx]

        img = nib.load(str(path))
        data = img.dataobj #type:ignore

        if len(img.shape) == 2: #type:ignore
            image = np.asarray(data, dtype=self.dtype)
        else:
            image = np.asarray(
                data[:, :, slice_idx],
                dtype=self.dtype,
            )

        if self.norm_image:
            low, high = np.percentile(image, [1, 99])
            image = np.clip(image, low, high)

            if high > low:
                image = (
                    2.0 * (image - low) / (high - low) - 1.0
                )
            else:
                image = np.zeros_like(image)

        image = np.ascontiguousarray(image[None, :, :])
        return torch.from_numpy(image)

# def normalize(tensor):
#     """
#     Normalize tensor from [0, 1] to [-1, 1]
    
#     Args:
#         tensor: Image tensor in range [0, 1]
#     Returns:
#         Normalized tensor in range [-1, 1]
#     """
#     return tensor * 2 - 1

# # Converting images back too 0,1 range so they may be saved and displayed correctly
# def denormalize(tensor):
#     """
#     Denormalize tensor from [-1, 1] to [0, 1]
    
#     Args:
#         tensor: Image tensor in range [-1, 1]
#     Returns:
#         Denormalized tensor in range [0, 1]
#     """
#     return (tensor + 1) / 2