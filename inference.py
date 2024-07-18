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

# Denormalize function
def denormalize(image, original_min, original_max):
    return image * (original_max - original_min) + original_min

# Load model function
def load_model(checkpoint_path, device):
    model = Generator(1, 1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Process image function
def process_image(image, transform):
    image = transform(image)
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: (1, 1, D, H, W)
    return image

# Sliding window inference function
def sliding_window_inference(model, image, patch_size, step_size, device):
    _, z, y, x = image.shape
    output = np.zeros((z, y, x))
    count_map = np.zeros((z, y, x))

    # Slide over the image with the patch size and step size
    for i in range(0, z - patch_size + 1, step_size):
        for j in range(0, y - patch_size + 1, step_size):
            for k in range(0, x - patch_size + 1, step_size):
                patch = image[:, i:i+patch_size, j:j+patch_size, k:k+patch_size]
                patch = torch.tensor(patch).to(device).float().unsqueeze(0)
                
                with torch.no_grad():
                    output_patch = model(patch)
                
                output_patch = output_patch.cpu().numpy()[0, 0]
                
                output[i:i+patch_size, j:j+patch_size, k:k+patch_size] += output_patch
                count_map[i:i+patch_size, j:j+patch_size, k:k+patch_size] += 1
    
    # Avoid division by zero
    count_map[count_map == 0] = 1
    output /= count_map

    return output

# Inference function
def infer(checkpoint_path, input_image_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)

    if input_image_path.endswith('.nii.gz'):
        img = nib.load(input_image_path)
        img_data = img.get_fdata()
        affine = img.affine
        header = img.header
    else:  # .nrrd
        img_data, header = nrrd.read(input_image_path)
        affine = np.eye(4)  # NRRD files do not have affine by default

    original_min = img_data.min()
    original_max = img_data.max()

    img_data = transform(img_data).numpy()  # Apply transform and convert to numpy array
    img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension: (1, D, H, W)

    patch_size = 128
    step_size = patch_size // 2  # Overlapping patches
    predicted_img_data = sliding_window_inference(model, img_data, patch_size, step_size, device)
    predicted_img_data = denormalize(predicted_img_data, original_min, original_max)

    # Prepare the output file path
    output_image_path = input_image_path.replace('.nii.gz', '_predicted_contrast.nii.gz').replace('.nrrd', '_predicted_contrast.nii.gz')
    
    if input_image_path.endswith('.nii.gz'):
        # Use the affine and header from the input image to save the output
        predicted_img = nib.Nifti1Image(predicted_img_data, affine, header)
        nib.save(predicted_img, output_image_path)
    else:  # .nrrd
        nrrd.write(output_image_path, predicted_img_data, header)

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