import torch
from torch import nn


def vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    hypers['batch_size'] = 4
    hypers['h_dim'] = 128
    hypers['z_dim'] = 64
    hypers['x_sigma2'] = 0.0001
    hypers['learn_rate'] = 0.000001
    hypers['betas'] = (0.85, 0.999)
    return hypers

# # Hyperparams
hp = vae_hyperparams()
batch_size = hp['batch_size']
h_dim = hp['h_dim']
z_dim = hp['z_dim']
x_sigma2 = hp['x_sigma2']
learn_rate = hp['learn_rate']
betas = hp['betas']


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(1306624, 64)
        self.fc2 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Encoded_Predictor(nn.Module):
    def __init__(self, modelA, modelB):
        super(Encoded_Predictor, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x1):
        x1 = torch.flatten(self.modelA.module.features_encoder(x1))
        x2 = self.modelB(x1)
        return x2