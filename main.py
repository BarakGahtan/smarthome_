import torch
import torch.optim as optim
from torch.nn import DataParallel

import auto_encoder
from data_prep import data_prep
import auto_encoder as autoencoder
from training import VAETrainer
from training import Trainer

VECTOR_LENGTH = 86400
def vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    hypers['batch_size'] = 4
    hypers['h_dim'] = 128
    hypers['z_dim'] = 64
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.85, 0.999)
    return hypers

# Hyperparams
hp = vae_hyperparams()
batch_size = hp['batch_size']
h_dim = hp['h_dim']
z_dim = hp['z_dim']
x_sigma2 = hp['x_sigma2']
learn_rate = hp['learn_rate']
betas = hp['betas']

# Data
dp = data_prep()
energy_train, energy_test = dp.getEnergy()
dl_sample = next(iter(energy_train))
print(dl_sample)  # Now the shape is 4,1,314


# Model
enc = auto_encoder.EncoderCNN()
dec = auto_encoder.DecoderCNN()
enc_feedforward = enc(dl_sample) # shape is 4,1024,319
decoded = dec(enc_feedforward) #shape is 4,1,314
print(enc_feedforward)


vae = autoencoder.VAE(encoder, decoder, 1, z_dim)
vae_dp = DataParallel(vae)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=learn_rate, betas=betas)

# Loss
def loss_fn(x, xr, z_mu, z_log_sigma2):
    return autoencoder.vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2)

# Trainer
trainer = VAETrainer(vae_dp, loss_fn, optimizer, device)