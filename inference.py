import torch
import nibabel as nib
import numpy as np
import os
from models import Generator
from torchvision import transforms
import logging
from skimage.util import view_as_windows
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
    c, z, y, x = image.shape
    window_shape = (c, patch_size, patch_size, patch_size)
    step_shape = (c, step_size, step_size, step_size)
    windows = view_as_windows(image, window_shape, step_shape)
    windows_shape = windows.shape
    windows = windows.reshape(-1, *window_shape)
    output_patches = []

    for patch in windows:
        patch = torch.tensor(patch).to(device).float()
        with torch.no_grad():
            output_patch = model(patch.unsqueeze(0))
        output_patches.append(output_patch.cpu().numpy()[0, 0])

    output_patches = np.array(output_patches)
    output_patches = output_patches.reshape(*windows_shape[:-4], *output_patches.shape[1:])
    output = np.zeros_like(image[0])

    count_map = np.zeros_like(output)
    for i in range(output_patches.shape[0]):
        for j in range(output_patches.shape[1]):
            for k in range(output_patches.shape[2]):
                output[i*step_size:i*step_size+patch_size, 
                       j*step_size:j*step_size+patch_size, 
                       k*step_size:k*step_size+patch_size] += output_patches[i, j, k]
                count_map[i*step_size:i*step_size+patch_size, 
                          j*step_size:j*step_size+patch_size, 
                          k*step_size:k*step_size+patch_size] += 1

    output /= count_map
    return output

# Inference function
def infer(checkpoint_path, input_image_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)

    if input_image_path.endswith('.nii.gz'):
        img = nib.load(input_image_path).get_fdata()
    else:  # .nrrd
        img, _ = nrrd.read(input_image_path)

    original_min = img.min()
    original_max = img.max()

    img = transform(img).numpy()  # Apply transform and convert to numpy array
    img = np.expand_dims(img, axis=0)  # Add channel dimension: (1, D, H, W)

    patch_size = 128
    step_size = patch_size // 2  # Overlapping patches
    predicted_img = sliding_window_inference(model, img, patch_size, step_size, device)
    predicted_img = denormalize(predicted_img, original_min, original_max)

    # Prepare the output file path
    output_image_path = input_image_path.replace('.nii.gz', '_predicted_contrast.nii.gz').replace('.nrrd', '_predicted_contrast.nii.gz')
    
    predicted_img_nifti = nib.Nifti1Image(predicted_img, np.eye(4))
    nib.save(predicted_img_nifti, output_image_path)
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