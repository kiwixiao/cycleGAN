import torch
from dataset import get_data_loaders
from models import Generator, Discriminator
from train import train
from evaluate import evaluate
from utils import logger

def main():
    logger.info("Starting 3D CycleGAN training process")

    # Hyperparameters
    batch_size = 1
    num_epochs = 200
    lr = 0.0002
    decay_epoch = 100

    # Paths
    noncontrast_dir = 'path/to/noncontrast/images'
    contrast_dir = 'path/to/contrast/images'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    noncontrast_loader, contrast_loader = get_data_loaders(noncontrast_dir, contrast_dir, batch_size)

    # Initialize models
    G_NC2C = Generator(1, 1).to(device)
    G_C2NC = Generator(1, 1).to(device)
    D_NC = Discriminator(1).to(device)
    D_C = Discriminator(1).to(device)

    logger.info("Models initialized and moved to device")

    # Train
    logger.info("Starting training")
    train(G_NC2C, G_C2NC, D_NC, D_C, noncontrast_loader, contrast_loader, num_epochs, device, lr, decay_epoch)

    # Evaluate
    logger.info("Starting evaluation")
    evaluate(G_NC2C, noncontrast_loader, device)

    logger.info("Process completed successfully")

if __name__ == '__main__':
    main()