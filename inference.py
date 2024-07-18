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
    for i in range(0, z, step_size):
        for j in range(0, y, step_size):
            for k in range(0, x, step_size):
                # Ensure patches fit within the image dimensions
                i_end = min(i + patch_size, z)
                j_end = min(j + patch_size, y)
                k_end = min(k + patch_size, x)
                
                # Create a patch of the expected size and pad if necessary
                patch = np.zeros((1, patch_size, patch_size, patch_size))
                patch[:, :i_end-i, :j_end-j, :k_end-k] = image[:, i:i_end, j:j_end, k:k_end]
                patch = torch.tensor(patch).to(device).float().unsqueeze(0)
                
                with torch.no_grad():
                    output_patch = model(patch)
                
                output_patch = output_patch.cpu().numpy()[0, 0, :i_end-i, :j_end-j, :k_end-k]
                
                output[i:i_end, j:j_end, k:k_end] += output_patch
                count_map[i:i_end, j:j_end, k:k_end] += 1
    
    # Avoid division by zero
    count_map[count_map == 0] = 1
    output /= count_map

    return output

# Inference function
def infer(checkpoint_path, input_image_path, transform, patch_size=128, step_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)

    if input_image_path.endswith('.nii.gz'):
        img = nib.load(input_image_path)
        img_data = img.get_fdata()
        affine = img.affine
        header = img.header
        original_dtype = img_data.dtype
    else:  # .nrrd
        img_data, header = nrrd.read(input_image_path)
        affine = None  # NRRD files do not have affine by default
        original_dtype = img_data.dtype

    original_min = img_data.min()
    original_max = img_data.max()

    img_data = transform(img_data).numpy()  # Apply transform and convert to numpy array
    img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension: (1, D, H, W)

    predicted_img_data = sliding_window_inference(model, img_data, patch_size, step_size, device)
    predicted_img_data = denormalize(predicted_img_data, original_min, original_max)
    predicted_img_data = predicted_img_data.astype(original_dtype)  # Ensure the data type matches the original

    # Prepare the output file path
    if input_image_path.endswith('.nii.gz'):
        output_image_path = input_image_path.replace('.nii.gz', '_predicted_contrast.nii.gz')
        predicted_img = nib.Nifti1Image(predicted_img_data, affine, header)
        nib.save(predicted_img, output_image_path)
    else:  # .nrrd
        output_image_path = input_image_path.replace('.nrrd', '_predicted_contrast.nrrd')
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