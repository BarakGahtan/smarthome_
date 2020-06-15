import dp as dp
import numpy as np
import torch.optim as optim
from torch.nn import DataParallel

import auto_encoder
import auto_encoder as autoencoder
from data_prep_aux import data_prep_aux as data_prep_Aux
from data_prep_therma_energy import data_prep_Energy, data_prep_Thermal
from training import VAETrainer

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
dp_energy = data_prep_Energy()
dp_thermal = data_prep_Thermal()
dp_aux = data_prep_Aux()
aux_data = dp_aux.returnDSAux()
thermal_data = dp_thermal.returnDSThermal()
energy_data = dp_energy.returnDSEnergy()
total_data = energy_data.D1_arrays_energy[0]
for i in range(1,len(energy_data.D1_arrays_energy)):
    total_data = np.append(total_data,energy_data.D1_arrays_energy[i])
for i in range(1,len(energy_data.D1_arrays_thermal)):
    total_data = np.append(total_data,energy_data.D1_arrays_thermal[i])
for i in range(0,len(aux_data)):
    total_data = np.append(total_data,aux_data[i],axis=0)

x=5

energy_train, energy_test = dp.getEnergy()
dl_sample = next(iter(energy_train))
print(dl_sample)  # Now the shape is 4,1,314


# Model
enc = auto_encoder.EncoderCNN()
dec = auto_encoder.DecoderCNN()
enc_feedforward = enc(dl_sample) # shape is 4,1024,319
decoded = dec(enc_feedforward) #shape is 4,1,314
print(enc_feedforward)


vae = autoencoder.VAE(enc, dec, 1, z_dim)
vae_dp = DataParallel(vae)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=learn_rate, betas=betas)

# Loss
def loss_fn(x, xr, z_mu, z_log_sigma2):
    return autoencoder.vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2)

# Trainer
trainer = VAETrainer(vae_dp, loss_fn, optimizer, device)