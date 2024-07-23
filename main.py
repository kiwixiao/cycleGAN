import torch
from dataset import get_data_loaders
from models import Generator, Discriminator, plot_model
from train import train
from evaluate import evaluate
from utils import logger

def estimate_model_memory(model, input_size, batch_size):
    input_tensor = torch.randn(batch_size, *input_size).cuda()
    output_tensor = model(input_tensor)
    
    input_memory = input_tensor.element_size() * input_tensor.nelement()
    output_memory = output_tensor.element_size() * output_tensor.nelement()
    param_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
    
    total_memory = input_memory + output_memory + param_memory
    return total_memory

def main():
    logger.info("Starting 3D CycleGAN training process")

    # Hyperparameters
    batch_size = 1
    num_epochs = 200
    lr = 0.0002
    decay_epoch = 100
    autosave_per_epochs = 20

    img_size = (1,128,128,128)
    memory_limit = 22 *1024**3

    # Paths
    noncontrast_dir = './CTNC'
    contrast_dir = './CTCE'
    test_noncontrast_dir = './TestCTNC'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Models initialized and using {device}")
    logger.info(f"Models initialized and using {device}")
    # Get data loaders
    noncontrast_loader, contrast_loader, test_noncontrast_loader = get_data_loaders(noncontrast_dir, contrast_dir, test_noncontrast_dir, batch_size)

    # Initialize models
    G_NC2C = Generator(1, 1).to(device)
    G_C2NC = Generator(1, 1).to(device)
    D_NC = Discriminator(1).to(device)
    D_C = Discriminator(1).to(device)

    logger.info("Models initialized and moved to device")
    # Save initial model structure
    plot_model(G_NC2C, torch.randn(1, 1, 128, 128, 128).to(device), 'G_NC2C_initial')
    plot_model(G_C2NC, torch.randn(1, 1, 128, 128, 128).to(device), 'G_C2NC_initial')
    plot_model(D_NC, torch.randn(1, 1, 128, 128, 128).to(device), 'D_NC_initial')
    plot_model(D_C, torch.randn(1, 1, 128, 128, 128).to(device), 'D_C_initial')

 # Estimate model memory
    try:
        total_memory_G_NC2C = estimate_model_memory(G_NC2C, img_size, batch_size)
        total_memory_G_C2NC = estimate_model_memory(G_C2NC, img_size, batch_size)
        total_memory_D_NC = estimate_model_memory(D_NC, img_size, batch_size)
        total_memory_D_C = estimate_model_memory(D_C, img_size, batch_size)

        total_memory = total_memory_G_NC2C + total_memory_G_C2NC + total_memory_D_NC + total_memory_D_C

        if total_memory > memory_limit:
            raise MemoryError(f"Estimated memory required: {total_memory / (1024**3):.2f} GB exceeds limit of 22 GB. Please reduce model size or batch size.")

        logger.info(f"Estimated total memory required: {total_memory / (1024**3):.2f} GB")
    except MemoryError as e:
        logger.error(str(e))
        return


    # Train
    logger.info("Starting training")
    train(G_NC2C, G_C2NC, D_NC, D_C, noncontrast_loader, contrast_loader, test_noncontrast_loader, num_epochs, device, lr, decay_epoch, autosave_per_epochs)

    # Evaluate
    logger.info("Starting evaluation")
    evaluate(G_NC2C, noncontrast_loader, device)

    logger.info("Process completed successfully")

    # after training is complete, delete dataloader
    del noncontrast_loader
    del contrast_loader
    del test_noncontrast_loader
    # clear GPU cache
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()