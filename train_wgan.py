import torch 
import torch.nn as nn
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from Data_Loader import *
from utils import*
from Model import *
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',       type   = int,          default = 200,          help = 'number of epochs of training')
parser.add_argument('--batch_size',     type   = int,          default = 5,            help = 'size of the batches'         )
parser.add_argument('--valid_split',    type   = float,        default = 0.1,          help = 'Validation split'            )
parser.add_argument('--lr',             type   = float,        default = 0.0002,       help = 'value of the learning rate'  )
parser.add_argument('--b1',             type   = float,        default = 0.5,          help = 'value of the beta 1'         )
parser.add_argument('--b2',             type   = float,        default = 0.999,        help = 'value of the beta 2'         )
parser.add_argument('--latent_dim',     type   = int,          default = 100,          help = 'latent space dimension (Z)'  )
parser.add_argument('--size',           type   = int,          default = 256,          help = 'image size'                  )
parser.add_argument('--channels',       type   = int,          default = 1,            help = 'image channels'              )
parser.add_argument('--n_critic',       type   = int,          default = 5,            help = 'value of n_critic'           )
parser.add_argument('--split_rate',     type   = float,        default = 0.8,          help = 'split rate'                  )
parser.add_argument('--lambda_gp',      type   = int,          default = 10,           help = 'lambda gp'                   )
parser.add_argument('--n_cpu',          type   = int,          default = 1,            help = 'number of CPU threads to use during batch generation')
parser.add_argument('--checkp_dir',     type   = str,          default = 'fit_model/', help = 'root directory to save the training checkpoints')
parser.add_argument('--data_root',      type   = str,          default = 'data/',      help = 'root directory of the dataset')
parser.add_argument('--exp_name',       type   = str,          default = 'exp1',       help = 'experiment name (numbers)'    )
opt = parser.parse_args()

# Create the checkpoint directory
directory_name = opt.checkp_dir + opt.exp_name+"/"
create_checkdirectory(directory_name)


# GAN Definition
img_shape        =  (opt.channels, opt.size, opt.size) 
G                =  Generator(img_shape, opt.latent_dim)
D                =  Discriminator(img_shape)


# Using GPU
device           =  torch.device('cuda:0')
G.to(device)
D.to(device)


# Optimizers and LR schedulers
optimizer_G      =  optim.Adam(G.parameters(), lr = opt.lr, weight_decay = 1e-5)
optimizer_D      =  optim.Adam(D.parameters(), lr = opt.lr, weight_decay = 1e-5)


# Split the dataset for training and validation
train_files, valid_files = split_path(opt.data_root, opt.valid_split, mode = "training")


# Transformations for the training dataset
transforms_ =   [   transforms.Resize( [int(opt.size),int(opt.size)], Image.BICUBIC ), 
                    transforms.RandomResizedCrop(int(opt.size), scale=(0.9, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5, resample=Image.BICUBIC,fill=(0,)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] )
                ]


# Dataloader for the training dataset
dataloader  = DataLoader(   My_Data( train_files, transforms_= transforms_, mode = 'training'), 
                            batch_size = opt.batch_size, shuffle = True, num_workers = opt.n_cpu )


# Transformations for the validation dataset
vtransforms_ =  [   transforms.Resize( [int(opt.size),int(opt.size)], Image.BICUBIC ), 
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] )  
                ]

# Dataloader for the validation dataset
dataloaderV = DataLoader(   My_Data(valid_files, transforms_= vtransforms_, mode = 'training'), 
                            batch_size = opt.batch_size, shuffle = True, num_workers = opt.n_cpu )


# Tensorboard definition for training and validation
tensorboard       = SummaryWriter('runs/WGAN_training/'   + opt.exp_name)
tensorboard_valid = SummaryWriter('runs/WGAN_validation/' + opt.exp_name)


# Criterio to evaluate the performance
criterion = nn.MSELoss()


# Establishing an initial best score
best_anom_score = 10000


# Function to compute the gradient penalty for WGAN
def compute_gradient_penalty(D, real_samples, fake_samples):

    # Computes the gradient penalty loss for WGAN GP
    # Random weight term for interpolation between real and fake samples
    alpha            =  torch.rand(*real_samples.shape[:2], 1, 1).to(device)

    # Get random interpolation between real and fake samples
    interpolates     =  (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates     =  autograd.Variable(interpolates, requires_grad = True)
    d_interpolates   =  D(interpolates)
    fake             =  torch.ones(*d_interpolates.shape).to(device)

    # Get gradients w.r.t. interpolates
    gradients        =  autograd.grad(outputs = d_interpolates, inputs = interpolates,
                              grad_outputs = fake, create_graph = True,
                              retain_graph = True, only_inputs = True)[0]
    
    gradients        =  gradients.view(gradients.shape[0], -1)
    gradient_penalty =  ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# Obtaining the epoch number to print
padding_epoch_t      =  len(str(opt.n_epochs))  
padding_t            =  len(str(len(dataloader)))

padding_epoch_v      =  len(str(opt.n_epochs))  
padding_v            =  len(str(len(dataloaderV)))


# ========================= TRAINING ===============================

for epoch in range(opt.n_epochs):

    for i, imgs in enumerate(dataloader):

        real_imgs    =  imgs.float()
        real_imgs    =  real_imgs.to(device)

        # --------------------------
        # Train Discriminator
        # --------------------------

        # Reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer_D.zero_grad()
        z = torch.randn(imgs.shape[0], opt.latent_dim).to(device) 

        # Generating a new sample
        fake_imgs        = G(z)
        real_validity    = D(real_imgs)
        fake_validity    = D(fake_imgs.detach())

        # Reconstruction Loss
        rec_loss = criterion(fake_imgs, real_imgs)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)

        # Adversarial loss
        d_loss           = (-torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty)
        d_loss.backward()

        optimizer_D.step()


        if i % opt.n_critic == 0:
            
            # ----------------------------------
            # Train Generator
            # ----------------------------------

            optimizer_G.zero_grad()
            fake_imgs     = G(z)
            fake_validity = D(fake_imgs)
            g_loss        = -torch.mean(fake_validity)
            
            # Updating weights
            g_loss.backward()
            optimizer_G.step()


            # Display the loss on tensorboard
            show_tensorboard(real_imgs[0], fake_imgs[0], [g_loss.item(), d_loss.item(), rec_loss.item()], epoch, tensorboard)

            # Save checkpoints
            if epoch % 10 == 0:
                torch.save(G.state_dict(),       os.path.join(directory_name, 'generator.pth'))
                torch.save(D.state_dict(),       os.path.join(directory_name, 'discriminator.pth'))

            # Printing loss 
            print(f"[Epoch {epoch:{padding_epoch_t}}/{opt.n_epochs}] "
                f"[Batch {i:{padding_t}}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():3f}] "
                f"[G loss: {g_loss.item():3f}]")


    # ------------- Validation --------------------------------------------------
    
    with torch.no_grad():

        for i, imgs in enumerate(dataloaderV):

            D.eval()
            G.eval()

            real_imgs_v   =  imgs.float()
            real_imgs_v   =  real_imgs_v.to(device)

            z = torch.randn(imgs.shape[0], opt.latent_dim).to(device)

            # Generating a new sample
            fake_imgs_v   = G(z)

            # Discriminator values
            real_validity = D(real_imgs_v)
            fake_validity = D(fake_imgs_v.detach())

            # Adversarial loss
            d_loss_v      = (-torch.mean(real_validity) + torch.mean(fake_validity) )#+ opt.lambda_gp * gradient_penalty)

            fake_imgs_v   = G(z)
            fake_validity = D(fake_imgs_v)

            g_loss_v      = -torch.mean(fake_validity)
            img_distance  = criterion(fake_imgs_v, real_imgs_v)
                
            # Display in tensorboard
            show_tensorboard_valid(real_imgs_v[0], fake_imgs_v[0], [g_loss_v.item(), d_loss_v.item(), img_distance.item()], epoch, tensorboard_valid)
                
            
            # Save checkpoints
            if img_distance < best_anom_score:
                torch.save(G.state_dict(),       os.path.join(directory_name, 'generator_best.pth'))
                torch.save(D.state_dict(),       os.path.join(directory_name, 'discriminator_best.pth'))
                best_anom_score = img_distance

                print("Save model with less error: ", best_anom_score, "--- epoch: ", (epoch+1))
                print("----------------------------------------")
                
                
            # Printing loss 
            print(f"[Epoch {epoch:{padding_epoch_v}}/{opt.n_epochs}] "
                f"[Batch {i:{padding_v}}/{len(dataloaderV)}] "
                f"[D loss: {d_loss_v.item():3f}] "
                f"[G loss: {g_loss_v.item():3f}]")
        # -------------------------------------------------

print("Finished Training - WGAN")