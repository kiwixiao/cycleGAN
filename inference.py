import torch
import nibabel as nib
import numpy as np
import os
from models import Generator
from torchvision import transforms
import logging
import nrrd
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import argparse



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
        # Normalize image to the range [0, 1]
        return 2 * ((image - image.min()) / (image.max() - image.min())) - 1

# Denormalize function
def denormalize(image, original_min, original_max):
    # Denormalize image back to the original intensity range
    return (image + 1) / 2 * (original_max - original_min) + original_min

# Load model function
def load_model(checkpoint_path, device):
    model = Generator(1, 1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Resample image function
def resample_image(image, new_spacing=[0.6, 0.6, 0.6], is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)

# Load image function
def load_image(file_path):
    if file_path.endswith('.nii.gz') or file_path.endswith('nii'):
        img = sitk.ReadImage(file_path)
    else:
        img, header = nrrd.read(file_path)
        img = sitk.GetImageFromArray(img)
        spacing = header.get('spacing', None) or [np.linalg.norm(direction) for direction in header.get('space directions', [])]
        if spacing:
            img.SetSpacing(spacing)
    return img
# Save image function
def save_image(image, original_image, file_path, postfix):
    resampled = resample_image(image, original_image.GetSpacing())
    if file_path.endswith('.nii.gz'):
        sitk.WriteImage(resampled, file_path.replace('.nii.gz', f'_{postfix}.nii.gz'))
    else:  # .nrrd
        img_array = sitk.GetArrayFromImage(resampled)
        header = {'spacings': original_image.GetSpacing()}
        nrrd.write(file_path.replace('.nrrd', f'_{postfix}.nrrd'), img_array, header)

def create_gaussian_window(size, sigma):
    """Create a Gaussian window."""
    coords = np.linspace(-1, 1, size)
    x, y, z = np.meshgrid(coords, coords, coords)
    gauss = np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    return gauss

# Sliding window inference function
def sliding_window_inference(model, image, patch_size, step_size, device, sigma=0.5):
    _, z, y, x = image.shape
    output = np.zeros((z, y, x))
    count_map = np.zeros((z, y, x))

    # create gaussian window
    gaussian_window = create_gaussian_window(patch_size, sigma)

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
                
                output[i:i_end, j:j_end, k:k_end] += output_patch * gaussian_window[:i_end-i, :j_end-j, :k_end-k]
                count_map[i:i_end, j:j_end, k:k_end] += gaussian_window[:i_end-i, :j_end-j, :k_end-k]
    
    # Avoid division by zero
    count_map[count_map == 0] = 1
    output /= count_map

    # apply Gaussian smoothing to the entire output
    #output = gaussian_filter(output, sigma=sigma)

    return output

# Inference function
def infer(checkpoint_path, input_image_path, transform, patch_size=128, step_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)

    original_image = load_image(input_image_path)
    resampled_image = resample_image(original_image)

    img_array = sitk.GetArrayFromImage(resampled_image)
    img_array = np.clip(img_array, -1000, 1000) # clip the HU between -1000 and 1000, this is how the model is trained as well.

    original_min = img_array.min()
    original_max = img_array.max()

    img_array = transform(img_array).numpy()
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Perform sliding window inference
    predicted_img_array = sliding_window_inference(model, img_tensor, patch_size, step_size, device)
    # Denormalize the predicted image data back to the original intensity range
    predicted_img_array = denormalize(predicted_img_array, original_min, original_max)
    
    predicted_img = sitk.GetImageFromArray(predicted_img_array)
    predicted_img.CopyInformation(resampled_image)

    save_image(resampled_image, original_image, input_image_path, 'resampled')
    save_image(predicted_img, original_image, input_image_path, 'fake_contrast')

    logger.info(f"Saved predicted fake contrast image to {input_image_path.replace('.nii.gz', '_fake_contrast.nii.gz').replace('.nrrd', '_fake_contrast.nrrd')}")

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Inference script for 3D CycleGAN")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("input_image_path", type=str, help="Path to the input 3D CT image")
    
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.from_numpy(img).float()),  # Convert to tensor
        Normalize(),  # Normalize the image data
    ])
    
    step_size = 32
    infer(args.checkpoint_path, args.input_image_path, transform)