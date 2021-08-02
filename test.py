import sys
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from Data_Loader import *
from Model import *
import argparse
from utils import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',  type   = int,          default = 1,                    help = 'size of the batches')
parser.add_argument('--latent_dim',  type   = int,          default = 100,                  help = 'latent space dimension (Z)')
parser.add_argument('--size',        type   = int,          default = 256,                  help = 'image size')
parser.add_argument('--channels',    type   = int,          default = 1,                    help = 'image channels')
parser.add_argument('--data_root',   type   = str,          default = 'data/',              help = 'root directory of the dataset')
parser.add_argument('--n_cpu',       type   = int,          default = 1,                    help = 'number of CPU threads to use during batch generation')
parser.add_argument('--net_G',       type   = str,          default = '/generator.pth',     help = 'generator network checkpoint file')
parser.add_argument('--net_D',       type   = str,          default = '/discriminator.pth', help = 'discriminator network checkpoint file')
parser.add_argument('--net_E',       type   = str,          default = '/encoder.pth',       help = 'encoder network checkpoint file')
parser.add_argument('--dir_model',   type   = str,          default = 'fit_model/',         help = 'directory that stores the fitted model')
parser.add_argument('--exp_name',    type   = str,          default = 'exp1',               help = 'experiment name (numbers)')
parser.add_argument('--dir_results', type   = str,          default = 'results/',           help = 'directory to save results'),
parser.add_argument('--best_model',  type   = bool,         default = False,                help = 'Selectio to save results of the best model'),
opt = parser.parse_args()


# Directories to read the models
dir_net_G   = opt.dir_model + opt.exp_name + opt.net_G
dir_net_D   = opt.dir_model + opt.exp_name + opt.net_D
dir_net_E   = opt.dir_model + opt.exp_name + opt.net_E


# Network definitions
img_shape = (opt.channels, opt.size, opt.size) 
G         = Generator(img_shape, opt.latent_dim)
D         = Discriminator(img_shape)
E         = Encoder(img_shape, opt.latent_dim)


# Using GPU
device = torch.device('cuda:0')
G.to(device)
D.to(device)
E.to(device)


# Load state dicts
G.load_state_dict(torch.load(dir_net_G))
D.load_state_dict(torch.load(dir_net_D))
E.load_state_dict(torch.load(dir_net_E))


# Transformation for the testing dataset
transforms_ = [ transforms.Resize( [int(opt.size),int(opt.size)], Image.BICUBIC ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
              ]


# Dataloader 
dataloader  = DataLoader(   My_Data (opt.data_root, transforms_ = transforms_, mode = 'testing'),
                            batch_size = opt.batch_size, shuffle = False, num_workers = opt.n_cpu
                        )


# Criterion to evaluate the performance
criterion = nn.MSELoss()


# Set model's test mode
G.eval()
D.eval()
E.eval()


# Lists to store results
list_input          = []
list_output         = []
anomaly_score_list  = []
img_distance_list   = []
z_distance_list     = []
kappa               = 1.0
batch_count         = 0


# ===================== Testing =================
for i, img in enumerate(dataloader):

    real_img      = img.float()
    real_img      = real_img.to(device)
    list_input.append(real_img[0])

    real_z        = E(real_img)
    fake_img      = G(real_z)
    list_output.append(fake_img[0])

    fake_z        = E(fake_img) 
    real_feature  = D.forward_features(real_img) 
    fake_feature  = D.forward_features(fake_img)
    real_feature  = real_feature / real_feature.max()
    fake_feature  = fake_feature / fake_feature.max()

    img_distance  = criterion(fake_img, real_img)
    loss_feature  = criterion(fake_feature, real_feature)
    anomaly_score = img_distance + kappa*loss_feature

    img_distance_list.append(img_distance.item())
    anomaly_score_list.append(anomaly_score.item())

    z_distance = criterion(fake_z, real_z)
    z_distance_list.append(z_distance.item())

    
    if batch_count == 0:
        z_real  = real_z
        z_fake  = fake_z
    
    else:
        z_real = torch.cat((z_real, real_z), dim=0)
        z_fake = torch.cat((z_fake, fake_z), dim=0)

    
    # Increase batch count
    batch_count += 1
    sys.stdout.write('\rGenerated images %d of %d' % ((i+1), len(dataloader)))

sys.stdout.write('\n')


# Directory to save results
if opt.best_model == False:
    directory_results = opt.dir_results + opt.exp_name

else:
    directory_results = opt.dir_results + opt.exp_name + "_best"


# Save results 
save_results(list_input, list_output, opt.size, directory_results)
save_scores_anom(img_distance_list, anomaly_score_list, z_distance_list, directory_results)


# Convert to numpy array and save the results
z_real = z_real.cpu().detach().numpy()
save_csv_z_real(z_real, directory_results)

z_fake = z_fake.cpu().detach().numpy()
save_csv_z_fake(z_fake, directory_results)