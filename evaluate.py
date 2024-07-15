import torch
import nibabel as nib
import numpy as np
from utils import logger, check_tensor_size

def evaluate(G_NC2C, noncontrast_loader, device, num_samples=10):
    G_NC2C.eval()
    with torch.no_grad():
        for i, noncontrast in enumerate(noncontrast_loader):
            if i >= num_samples:
                break
            check_tensor_size(noncontrast, (1, 1, 128, 128, 128), f"Evaluation input {i}")
            noncontrast = noncontrast.to(device).half()
            fake_contrast = G_NC2C(noncontrast)
            check_tensor_size(fake_contrast, (1, 1, 128, 128, 128), f"Evaluation output {i}")
            
            fake_contrast_np = fake_contrast.cpu().numpy()
            nifti_img = nib.Nifti1Image(fake_contrast_np, np.eye(4))
            nib.save(nifti_img, f'fake_contrast_{i}.nii.gz')

        logger.info(f"Evaluation completed. {num_samples} samples processed.")