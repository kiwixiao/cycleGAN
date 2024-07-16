import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from utils import logger, check_tensor_size
import matplotlib.pyplot as plt

def plot_and_save(training_losses, title, ylabel, filename):
    plt.figure()
    plt.plot(training_losses, label='Training')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def train(G_NC2C, G_C2NC, D_NC, D_C, noncontrast_loader, contrast_loader, num_epochs, device, lr=0.0002, decay_epoch=100):
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(list(G_NC2C.parameters()) + list(G_C2NC.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_NC = optim.Adam(D_NC.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_C = optim.Adam(D_C.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - decay_epoch) / float(num_epochs - decay_epoch))
    lr_scheduler_D_NC = optim.lr_scheduler.LambdaLR(optimizer_D_NC, lr_lambda=lambda epoch: 1.0 - max(0, epoch - decay_epoch) / float(num_epochs - decay_epoch))
    lr_scheduler_D_C = optim.lr_scheduler.LambdaLR(optimizer_D_C, lr_lambda=lambda epoch: 1.0 - max(0, epoch - decay_epoch) / float(num_epochs - decay_epoch))

    #scaler = GradScaler() # this is for auto graph, comment out for double precision
    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        G_epoch_loss = 0
        D_epoch_loss = 0
        for i, (noncontrast, contrast) in enumerate(zip(noncontrast_loader, contrast_loader)):
            check_tensor_size(noncontrast, (noncontrast.size(0), 1, 128, 128, 128), "Noncontrast input")
            check_tensor_size(contrast, (contrast.size(0), 1, 128, 128, 128), "Contrast input")

            noncontrast = noncontrast.to(device).float() # convert input to float16 to match autocase mix precision.
            contrast = contrast.to(device).float()

            valid = torch.ones((noncontrast.size(0), 1, 8, 8, 8), requires_grad=False).to(device).float()
            fake = torch.zeros((noncontrast.size(0), 1, 8, 8, 8), requires_grad=False).to(device).float())

            # Train Generators
            optimizer_G.zero_grad()

            #with autocast():
            loss_id_NC = criterion_identity(G_C2NC(noncontrast), noncontrast)
            loss_id_C = criterion_identity(G_NC2C(contrast), contrast)
            loss_identity = (loss_id_NC + loss_id_C) / 2

            fake_contrast = G_NC2C(noncontrast)
            loss_GAN_NC2C = criterion_GAN(D_C(fake_contrast), valid)

            fake_noncontrast = G_C2NC(contrast)
            loss_GAN_C2NC = criterion_GAN(D_NC(fake_noncontrast), valid)

            loss_GAN = (loss_GAN_NC2C + loss_GAN_C2NC) / 2

            recovered_noncontrast = G_C2NC(fake_contrast)
            loss_cycle_NC = criterion_cycle(recovered_noncontrast, noncontrast)

            recovered_contrast = G_NC2C(fake_noncontrast)
            loss_cycle_C = criterion_cycle(recovered_contrast, contrast)

            loss_cycle = (loss_cycle_NC + loss_cycle_C) / 2

            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity

            #scaler.scale(loss_G).backward()
            #scaler.step(optimizer_G)
            loss_G.backward()
            optimizer_G.step()
            # Train Discriminators
            optimizer_D_NC.zero_grad()
            optimizer_D_C.zero_grad()

            #with autocast():
            loss_real_NC = criterion_GAN(D_NC(noncontrast), valid)
            loss_real_C = criterion_GAN(D_C(contrast), valid)
            
            loss_fake_NC = criterion_GAN(D_NC(fake_noncontrast.detach()), fake)
            loss_fake_C = criterion_GAN(D_C(fake_contrast.detach()), fake)
            
            loss_D_NC = (loss_real_NC + loss_fake_NC) / 2
            loss_D_C = (loss_real_C + loss_fake_C) / 2
            loss_D = (loss_D_NC + loss_D_C) / 2

            # scaler.scale(loss_D).backward()
            # scaler.step(optimizer_D_NC)
            # scaler.step(optimizer_D_C)
            loss_D.backward()
            optimizer_D_NC.step()
            optimizer_D_C.step()

            # scaler.update()
            G_epoch_loss += loss_G.item()
            D_epoch_loss += loss_D.item()

            if i % 100 == 0:
                logger.info(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(noncontrast_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        lr_scheduler_G.step()
        lr_scheduler_D_NC.step()
        lr_scheduler_D_C.step()

        if (epoch+1) % 50 == 0:
            logger.info(f"Saving models at epoch {epoch+1}")
            torch.save(G_NC2C.state_dict(), f'G_NC2C_{epoch+1}.pth')
            torch.save(G_C2NC.state_dict(), f'G_C2NC_{epoch+1}.pth')
            torch.save(D_NC.state_dict(), f'D_NC_{epoch+1}.pth')
            torch.save(D_C.state_dict(), f'D_C_{epoch+1}.pth')

    # Plot and save loss graphs
    plot_and_save(G_losses, 'Generator Loss', 'Loss', 'generator_loss.png')
    plot_and_save(D_losses, 'Discriminator Loss', 'Loss', 'discriminator_loss.png')