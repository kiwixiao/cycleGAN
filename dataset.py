# Add at the top of the file
from utils import logger, check_tensor_size

class CTDataset(Dataset):
    def __getitem__(self, idx):
        # ... (existing code) ...
        
        if self.transform:
            img = self.transform(img)
        
        check_tensor_size(img, (1, self.patch_size, self.patch_size, self.patch_size), f"Dataset item {idx}")
        return img

def get_data_loaders(noncontrast_dir, contrast_dir, batch_size):
    # ... (existing code) ...

    logger.info(f"Noncontrast dataset size: {len(noncontrast_dataset)}")
    logger.info(f"Contrast dataset size: {len(contrast_dataset)}")

    return noncontrast_loader, contrast_loader