import os

import numpy as np
import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import random_split, DataLoader

import auto_encoder
import auto_encoder as autoencoder
import main_data as dataLoader
from training import VAETrainer
from utils import plot
from utils.plot import plot_fit, plot_accuracy

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# preparing different datasets
# 1 - aux sensor only
data_aux = dataLoader.DatasetCombined(1)
data_aux_list = []
for i in range(len(data_aux.data)):
    mat = np.expand_dims(data_aux.data[i].numpy(), axis=0)
    data_aux_list.append(mat)
split_lengths_aux = [int(len(data_aux_list)*0.6+1), int(len(data_aux_list)*0.4)]
ds_train_aux, ds_test_aux = random_split(data_aux_list, split_lengths_aux)
dl_train_aux = DataLoader(ds_train_aux, batch_size=4, shuffle=True)
dl_test_aux = DataLoader(ds_test_aux, batch_size=4, shuffle=True)
x0 = data_aux_list[0]
raw_sample = torch.from_numpy(x0).float().to(device)
dataload_sample = next(iter(dl_train_aux))

# 2 - thermal sensor only
data_thermal, thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(2)
data_thermal = data_thermal.squeeze(0)
data_thermal_list = []
for i in range(np.shape(data_thermal)[0]):
    mat = np.expand_dims(data_thermal[i], axis=0)
    data_thermal_list.append(mat)

split_lengths_thermal = [int(len(data_thermal_list)*0.6+1), int(len(data_thermal_list)*0.4)]
ds_train_thermal, ds_test_thermal = random_split(data_thermal_list, split_lengths_thermal)
dl_train_thermal = DataLoader(ds_train_thermal, batch_size=4, shuffle=True)
dl_test_thermal = DataLoader(ds_test_thermal, batch_size=4, shuffle=True)

# 3 - energy sensor only
data_energy,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(3)
data_energy = data_energy.squeeze(0)
data_energy_list = []
for i in range(np.shape(data_energy)[0]):
    mat = np.expand_dims(data_energy[i], axis=0)
    data_energy_list.append(mat)
split_lengths_energy = [int(len(data_energy_list)*0.6+1), int(len(data_energy_list)*0.4)]
ds_train_energy, ds_test_energy = random_split(data_energy_list, split_lengths_energy)
dl_train_energy = DataLoader(ds_train_energy, batch_size=4, shuffle=True)
dl_test_energy = DataLoader(ds_test_energy, batch_size=4, shuffle=True)

# 4 - thermal and aux
data_thermal_aux = data_thermal_list + data_aux_list
split_lengths_thermal_aux = [int(len(data_thermal_aux)*0.6), int(len(data_thermal_aux)*0.4)]
ds_train_thermal_aux, ds_test_thermal_aux = random_split(data_thermal_aux, split_lengths_thermal_aux)
dl_train_thermal_aux = DataLoader(ds_train_thermal_aux, batch_size=4, shuffle=True)
dl_test_thermal_aux = DataLoader(ds_test_thermal_aux, batch_size=4, shuffle=True)

# 5 - energy and aux
data_energy_aux = data_energy_list+ data_aux_list
split_lengths_energy_aux = [int(len(data_energy_aux)*0.6), int(len(data_energy_aux)*0.4)]
ds_train_energy_aux, ds_test_energy_aux = random_split(data_energy_aux, split_lengths_energy_aux)
dl_train_energy_aux = DataLoader(ds_train_energy_aux, batch_size=4, shuffle=True)
dl_test_energy_aux = DataLoader(ds_test_energy_aux, batch_size=4, shuffle=True)

# 6 - energy and thermal
data_energy_thermal = data_energy_list + data_thermal_list
split_lengths_energy_thermal = [int(len(data_energy_thermal)*0.6+1), int(len(data_energy_thermal)*0.4)]
ds_train_energy_thermal, ds_test_energy_thermal = random_split(data_energy_thermal, split_lengths_energy_thermal)
dl_train_energy_thermal = DataLoader(ds_train_energy_thermal, batch_size=4, shuffle=True)
dl_test_energy_thermal = DataLoader(ds_test_energy_thermal, batch_size=4, shuffle=True)

# 7 - energy, thermal and aux
data_energy_thermal_aux = data_energy_list + data_thermal_list + data_aux_list
split_lengths_energy_thermal_aux = [int(len(data_energy_thermal_aux)*0.6+1), int(len(data_energy_thermal_aux)*0.4)]
ds_train_energy_thermal_aux, ds_test_energy_thermal_aux = random_split(data_energy_thermal_aux, split_lengths_energy_thermal_aux)
dl_train_energy_thermal_aux = DataLoader(ds_train_energy_thermal_aux, batch_size=4, shuffle=True)
dl_test_energy_thermal_aux = DataLoader(ds_test_energy_thermal_aux, batch_size=4, shuffle=True)

# Model
encoder = auto_encoder.EncoderCNN()
decoder = auto_encoder.DecoderCNN()
#raw_sample before dataloader shape is [1,314]
#dataloader_sample after DataLoader is [4,1,314]
#encoder_output sample size is [4,1024,319]
#decoder_output sample size is [4,1,314]

#check per one
enc_feedforward =  encoder(dataload_sample)# shape should be 4,1024,319, should it be z?
decoded = decoder(enc_feedforward) #shape is 3,1,314 ( back to normal)
vae = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)#(features_encoder, features_decoder, in_size, z_dim)
vae_dp = DataParallel(vae)

i = 0
list_DL_tuples = []
list_DL_tuples.append( (dl_train_aux, dl_test_aux)) #0
list_DL_tuples.append( (dl_train_thermal, dl_test_thermal)) #1
list_DL_tuples.append( (dl_train_energy, dl_test_energy)) #2
list_DL_tuples.append( (dl_train_thermal_aux, dl_test_thermal_aux)) #3
list_DL_tuples.append( (dl_train_energy_aux, dl_test_energy_aux)) #4
list_DL_tuples.append( (dl_train_energy_thermal, dl_test_energy_thermal)) #5
list_DL_tuples.append( (dl_train_energy_thermal_aux, dl_test_energy_thermal_aux)) #6

# Loss
def loss_fn_mse(x, xr, z_mu, z_log_sigma2):
    return autoencoder.vae_loss_mse(x, xr, z_mu, z_log_sigma2, x_sigma2)

def loss_fn_CE(x, xr, z_mu, z_log_sigma2):
    return autoencoder.vae_loss_CE(x, xr, z_mu, z_log_sigma2, x_sigma2)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=learn_rate, betas=betas)

# Trainer
trainer_mse = VAETrainer(vae_dp, loss_fn_mse, optimizer, device)
trainer_CE = VAETrainer(vae_dp, loss_fn_CE, optimizer, device)

# test_vae_loss()
checkpoint_file = 'checkpoints/vae'
checkpoint_file_final = f'{checkpoint_file}_MSE_latentSpace'
if os.path.isfile(f'{checkpoint_file}.pt'):
    os.remove(f'{checkpoint_file}.pt')

if os.path.isfile(f'{checkpoint_file_final}.pt'):
    print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
    checkpoint_file = checkpoint_file_final
else:
    fit_results = []
    for i in range(6,len(list_DL_tuples)):
        current_check_point_file = f'{checkpoint_file_final}_' + str(i)
        res = trainer_mse.fit(list_DL_tuples[i][0], list_DL_tuples[i][1],
                              num_epochs=200, early_stopping=20, print_every=10,
                              checkpoints=current_check_point_file,
                              post_epoch_fn=None)
        plot_accuracy(res, "model " +str(i) + " latent space - accuracy", True)
        # plot_accuracy(res, "model " +str(i) + " latent space - average loss", False)

