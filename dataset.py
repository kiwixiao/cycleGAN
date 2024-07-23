import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import nrrd
from torchvision import transforms
from utils import logger, check_tensor_size
import numpy as np
import SimpleITK as sitk

def resample_image(image, new_spacing=[0.6, 0.6, 0.6], is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2]))) 
    ]

    resample = sitk.ResampleImageFilter() # initilize an instance
    resample.SetOutputSpacing(new_spacing) # call method
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection()) # make sure resampled image is the same orientation.
    resample.SetOutputOrigin(image.GetOrigin)
    resample.SetTransform(sitk.Transform()) # apply an identity transformation for ressampling
    resample.SetDefaultPixelValue(image.GetPixelIDValue()) # give pixel an default value if resample pixel is outside the original image space
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.Linear)

    return resample.Execute(image)

class Normalize:
    def __call__(self, image):
        # normalize it between -1 and 1, which is more common in cycleGAN network
        return 2* ((image - image.min()) / (image.max() - image.min())) -1


class CTDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=128):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.nii.gz') or f.endswith('.nrrd')]
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        if file_path.endswith('.nii.gz'):
            img = nib.load(file_path).get_fdata()
            img = sitk.GetImageFromArray(img)
        else:  # .nrrd
            img, header = nrrd.read(file_path)
            img = sitk.GetImageFromArray(img)
            spacing = header.get('spacings', header.get('space directions', None))
            if spacing is not None:
                img.SetSpacing(spacing)
        
        img = resample_image(img)
        img = sitk.GetArrayFromImage(img)
        # clip the image intensity between -1000 and 1000 HU
        img = np.clip(img, -1000, 1000) # this is optional, if the training is not optimial, maybe can comment out.

        if self.patch_size:
            # Extract a random patch
            x = random.randint(0, img.shape[0] - self.patch_size)
            y = random.randint(0, img.shape[1] - self.patch_size)
            z = random.randint(0, img.shape[2] - self.patch_size)
            img = img[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
            
        if self.transform:
            img = self.transform(img)
        #add channel dimension: (1, 128, 128, 128)
        img = img.unsqueeze(0)
        check_tensor_size(img, (1, self.patch_size, self.patch_size, self.patch_size), f"Dataset item {idx}")
        
        return img

def get_data_loaders(noncontrast_dir, contrast_dir, test_noncontrast_dir, batch_size, patch_size=64):
    transform = transforms.Compose([
        Normalize(),
        transforms.ToTensor(),
    ])
    
    noncontrast_dataset = CTDataset(root_dir=noncontrast_dir, transform=transform, patch_size=patch_size)
    contrast_dataset = CTDataset(root_dir=contrast_dir, transform=transform, patch_size=patch_size)

    test_noncontrast_dataset = CTDataset(root_dir=test_noncontrast_dir, transform=transform, patch_size=patch_size)

    logger.info(f"Noncontrast dataset size: {len(noncontrast_dataset)}")
    logger.info(f"Contrast dataset size: {len(contrast_dataset)}")

    noncontrast_loader = DataLoader(noncontrast_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False) # add num_workers and pin_memory to keep gpu busy all the time. this allows more efficient data feed from cpu to gpu.
    contrast_loader = DataLoader(contrast_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

    test_noncontrast_loader = DataLoader(test_noncontrast_dataset, batch_size=batch_size, shuffle=False)

    return noncontrast_loader, contrast_loader, test_noncontrast_loader