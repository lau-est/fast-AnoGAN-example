import torch 
import torch.nn as nn
from torch import autograd, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from Data_Loader import *
from utils import*
from Model import *
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',       type   = int,          default = 16,           help = 'number of epochs of training')
parser.add_argument('--batch_size',     type   = int,          default = 4,            help = 'size of the batches'         )
parser.add_argument('--valid_split',    type   = float,        default = 0.1,          help = 'Validation split')
parser.add_argument('--lr',             type   = float,        default = 0.0002,       help = 'value of the learning rate'  )
parser.add_argument('--b1',             type   = float,        default = 0.5,          help = 'value of the beta 1'         )
parser.add_argument('--b2',             type   = float,        default = 0.999,        help = 'value of the beta 2'         )
parser.add_argument('--latent_dim',     type   = int,          default = 100,          help = 'latent space dimension (Z)'  )
parser.add_argument('--size',           type   = int,          default = 256,          help = 'image size'                  )
parser.add_argument('--channels',       type   = int,          default = 1,            help = 'image channels'              )
parser.add_argument('--n_critic',       type   = int,          default = 5,            help = 'value of n_critic'           )
parser.add_argument('--n_cpu',          type   = int,          default = 1,            help = 'number of CPU threads to use during batch generation')
parser.add_argument('--checkp_dir',     type   = str,          default = 'fit_model/', help = 'root directory to save the training checkpoints')
parser.add_argument('--data_root',      type   = str,          default = 'data/',      help = 'root directory of the dataset')
parser.add_argument('--exp_name',       type   = str,          default = 'exp1',       help = 'experiment name (numbers)'   )
opt = parser.parse_args()


# Create the checkpoint directory
directory_name = opt.checkp_dir + opt.exp_name+"/"
create_checkdirectory(directory_name)


# WGAN Network Definition
img_shape        =  (opt.channels, opt.size, opt.size)
G                =  Generator(img_shape, opt.latent_dim)
D                =  Discriminator(img_shape)


# Load state dicts from the WGAN model
path_G = directory_name + "generator.pth"
path_D = directory_name + "discriminator.pth"
G.load_state_dict(torch.load(path_G))
D.load_state_dict(torch.load(path_D))


# Encoder Network Definition
E             = Encoder(img_shape, opt.latent_dim)


# Using GPU
device           =  torch.device('cuda:0')
G.to(device)
D.to(device)
E.to(device)


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
tensorboard   = SummaryWriter('runs/Encoder_training/'   + opt.exp_name)
tensorboardV  = SummaryWriter("runs/Encoder_validation/" + opt.exp_name)


# Loss Function or riterio to evaluate the performance
criterion     = nn.MSELoss()
optimizer_E   = torch.optim.Adam(E.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))


# Obtaining the epoch number to print
padding_epoch_t = len(str(opt.n_epochs))
padding_t       = len(str(len(dataloader)))
padding_epoch_v = len(str(opt.n_epochs))
padding_v       = len(str(len(dataloaderV)))
kappa           = 1.0


# Establishing an initial best score
best_anom_score = 10000


# Performance evaluation of the generator and discriminotor network
G.eval()
D.eval()


# ================ ENCODER TRAINING =====================
for epoch in range(opt.n_epochs):

    for i, imgs in enumerate(dataloader):
        
        real_imgs     =  imgs.float()
        real_imgs     =  real_imgs.to(device)

        optimizer_E.zero_grad()
        z             = E(real_imgs) 
        fake_imgs     = G(z)

        # Image features
        real_features = D.forward_features(real_imgs)
        fake_features = D.forward_features(fake_imgs)


        # izif architecture
        loss_imgs      = criterion(fake_imgs, real_imgs)
        loss_features  = criterion(fake_features, real_features)
        e_loss         = loss_imgs + kappa * loss_features

        e_loss.backward()
        optimizer_E.step()


        if i % opt.n_critic == 0:
           
            # Display in tensorboard
            show_tensorboard_enc(real_imgs[0], fake_imgs[0], [e_loss.item()], epoch, tensorboard)

            # Save checkpoints
            if epoch % 10 == 0:
                torch.save(E.state_dict(), os.path.join(directory_name, 'encoder.pth'))
           
            print(f"[Epoch {epoch:{padding_epoch_t}}/{opt.n_epochs}] "
                    f"[Batch {i:{padding_t}}/{len(dataloader)}] "
                    f"[E loss: {e_loss.item():3f}]")


    # -------------------------- VALIDATION -------------------------------
    with torch.no_grad():

        for i, imgs in enumerate(dataloaderV):
        
            E.eval()

            real_imgs_v    =  imgs.float()
            real_imgs_v    =  real_imgs_v.to(device)
            z              = E(real_imgs_v) 
            fake_imgs_v    = G(z)


            # Image features
            real_features_v = D.forward_features(real_imgs_v)
            fake_features_v = D.forward_features(fake_imgs_v)


            # izif architecture
            loss_imgs      = criterion(fake_imgs_v, real_imgs_v)
            loss_features  = criterion(fake_features_v, real_features_v)
            e_loss_v       = loss_imgs + kappa * loss_features


            if i % opt.n_critic == 0:
            
                # Display in tensorboard
                show_tensorboard_enc_val(real_imgs_v[0], fake_imgs_v[0], [e_loss_v.item()], epoch, tensorboard)

                # Save checkpoints
                if e_loss_v.item() < best_anom_score:
                    torch.save(E.state_dict(), os.path.join(directory_name, 'encoder_best.pth'))
                    best_anom_score = e_loss_v.item()

                    print("Save model with less error: ", best_anom_score, "--- epoch: ", (epoch+1))
                    print("----------------------------------------")
            
                print(f"[Epoch {epoch:{padding_epoch_v}}/{opt.n_epochs}] "
                        f"[Batch {i:{padding_v}}/{len(dataloaderV)}] "
                        f"[E loss: {e_loss_v.item():3f}]")
            

print("Finished Encoder Traininng")