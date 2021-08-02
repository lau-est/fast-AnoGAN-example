import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feature, out_feature, normalize=True):
            layers = [nn.Linear(in_feature, out_feature)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feature, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, input):
        img = self.model(input)
        
        return img.view(img.shape[0], *self.img_shape)




class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward_features(self, img):
        img_flat = img.view(img.shape[0], -1) 
        features = self.features(img_flat)
        return features

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.last_layer(features)
        return validity



class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity