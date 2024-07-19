import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import nrrd
from torchvision import transforms
from utils import logger, check_tensor_size
import numpy as np

class Normalize(object):
    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


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
        else:  # .nrrd
            img, _ = nrrd.read(file_path)
        
        # clip the image intensity between -1000 and 1000 HU
        img = np.clip(img, -1000, 1000) # this is optional, if the training is not optimial, maybe can comment out.

        original_min = img.min()
        original_max = img.max()
        
        if self.patch_size:
            # Extract a random patch
            x = random.randint(0, img.shape[0] - self.patch_size)
            y = random.randint(0, img.shape[1] - self.patch_size)
            z = random.randint(0, img.shape[2] - self.patch_size)
            img = img[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
            
            expected_size = (1, self.patch_size, self.patch_size, self.patch_size)
        else:
            expected_size = (1, *img.shape)    

        if self.transform:
            img = self.transform(img)
        #add channel dimension: (1, 128, 128, 128)
        img = img.unsqueeze(0)
        
        check_tensor_size(img, (1, self.patch_size, self.patch_size, self.patch_size), f"Dataset item {idx}")
        return img

def get_data_loaders(noncontrast_dir, contrast_dir, test_noncontrast_dir, batch_size, patch_size=128):
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