import torch
import nibabel as nib
import numpy as np
import os
from models import Generator
from torchvision import transforms
import logging
import nrrd

# Set up logger
def setup_logger():
    logger = logging.getLogger("InferenceLogger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

# Normalize class
class Normalize(object):
    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())

def denormalize(image, original_min, original_max):
    return image * (original_max - original_min) + original_min

def load_model(checkpoint_path, device):
    model = Generator(1, 1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def process_image(image, transform):
    image = transform(image)
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: (1, 1, D, H, W)
    return image

def infer(checkpoint_path, input_image_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)

    if input_image_path.endswith('.nii.gz'):
        img = nib.load(input_image_path).get_fdata()
    else:  # .nrrd
        img, _ = nrrd.read(input_image_path)

    original_min = img.min()
    original_max = img.max()

    img = process_image(img, transform)
    img = img.to(device).float()

    with torch.no_grad():
        fake_contrast = model(img)
    
    fake_contrast_np = fake_contrast.cpu().numpy()[0, 0, :, :, :]
    fake_contrast_np = denormalize(fake_contrast_np, original_min, original_max)

    # Prepare the output file path
    output_image_path = input_image_path.replace('.nii.gz', '_predicted_contrast.nii.gz').replace('.nrrd', '_predicted_contrast.nii.gz')
    
    fake_contrast_img = nib.Nifti1Image(fake_contrast_np, np.eye(4))
    nib.save(fake_contrast_img, output_image_path)
    logger.info(f"Saved predicted fake contrast image to {output_image_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for 3D CycleGAN")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("input_image_path", type=str, help="Path to the input 3D CT image")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        Normalize(),
    ])

    infer(args.checkpoint_path, args.input_image_path, transform)