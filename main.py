import os
import IPython.display
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import auto_encoder
import auto_encoder as autoencoder
import main_data as dataLoader
from cs236781 import plot
from training import VAETrainer
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

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
    hypers['learn_rate'] = 0.00001
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

#preparing different datasets
# 1 - aux sensor only
data_aux = dataLoader.DatasetCombined(1)
split_lengths_aux = [int(len(data_aux)*0.6+1), int(len(data_aux)*0.4)]
ds_train_aux, ds_test_aux = random_split(data_aux, split_lengths_aux)
dl_train_aux = DataLoader(ds_train_aux, batch_size=4, shuffle=True)
dl_test_aux = DataLoader(ds_test_aux, batch_size=4, shuffle=True)

# 2 - thermal sensor only
data_thermal = dataLoader.DatasetCombined(2)
split_lengths_thermal = [int(len(data_thermal)*0.6+1), int(len(data_thermal)*0.4)]
ds_train_thermal, ds_test_thermal = random_split(data_thermal, split_lengths_thermal)
dl_train_thermal = DataLoader(ds_train_thermal, batch_size=4, shuffle=True)
dl_test_thermal = DataLoader(ds_test_thermal, batch_size=4, shuffle=True)

# 3 - energy sensor only
data_energy = dataLoader.DatasetCombined(3)
split_lengths_energy = [int(len(data_energy)*0.6+1), int(len(data_energy)*0.4)]
ds_train_energy, ds_test_energy = random_split(data_energy, split_lengths_energy)
dl_train_energy = DataLoader(ds_train_energy, batch_size=4, shuffle=True)
dl_test_energy = DataLoader(ds_test_energy, batch_size=4, shuffle=True)

# 4 - thermal and aux
data_thermal_aux = dataLoader.DatasetCombined(4)
split_lengths_thermal_aux = [int(len(data_thermal_aux)*0.6+1), int(len(data_thermal_aux)*0.4)]
ds_train_thermal_aux, ds_test_thermal_aux = random_split(data_thermal_aux, split_lengths_thermal_aux)
dl_train_thermal_aux = DataLoader(ds_train_thermal_aux, batch_size=4, shuffle=True)
dl_test_thermal_aux = DataLoader(ds_test_thermal_aux, batch_size=4, shuffle=True)

# 5 - energy and aux
data_energy_aux = dataLoader.DatasetCombined(5)
split_lengths_energy_aux = [int(len(data_energy_aux)*0.6+1), int(len(data_energy_aux)*0.4)]
ds_train_energy_aux, ds_test_energy_aux = random_split(data_energy_aux, split_lengths_energy_aux)
dl_train_energy_aux = DataLoader(ds_train_energy_aux, batch_size=4, shuffle=True)
dl_test_energy_aux = DataLoader(ds_test_energy_aux, batch_size=4, shuffle=True)

# 6 - energy and thermal
data_energy_thermal = dataLoader.DatasetCombined(6)
split_lengths_energy_thermal = [int(len(data_energy_thermal)*0.6+1), int(len(data_energy_thermal)*0.4)]
ds_train_energy_thermal, ds_test_energy_thermal = random_split(data_energy_thermal, split_lengths_energy_thermal)
dl_train_energy_thermal = DataLoader(ds_train_energy_thermal, batch_size=4, shuffle=True)
dl_test_energy_thermal = DataLoader(ds_test_energy_thermal, batch_size=4, shuffle=True)

# 7 - energy, thermal and aux
data_energy_thermal_aux = dataLoader.DatasetCombined(7)
split_lengths_energy_thermal_aux = [int(len(data_energy_thermal_aux)*0.6+1), int(len(data_energy_thermal_aux)*0.4)]
ds_train_energy_thermal_aux, ds_test_energy_thermal_aux = random_split(data_energy_thermal_aux, split_lengths_energy_thermal_aux)
dl_train_energy_thermal_aux = DataLoader(ds_train_energy_thermal_aux, batch_size=4, shuffle=True)
dl_test_energy_thermal_aux = DataLoader(ds_test_energy_thermal_aux, batch_size=4, shuffle=True)

i = 0
list_DL_tuples = []
list_DL_tuples.append( (dl_train_aux, dl_test_aux))
list_DL_tuples.append( (dl_train_thermal, dl_test_thermal))
list_DL_tuples.append( (dl_train_energy, dl_test_energy))
list_DL_tuples.append( (dl_train_thermal_aux, dl_test_thermal_aux))
list_DL_tuples.append( (dl_train_energy_aux, dl_test_energy_aux))
list_DL_tuples.append( (dl_train_energy_thermal, dl_test_energy_thermal))
list_DL_tuples.append( (dl_train_energy_thermal_aux, dl_test_energy_thermal_aux))

# Model
encoder = auto_encoder.EncoderCNN()
decoder = auto_encoder.DecoderCNN()





#Data
data = dataLoader.DatasetCombined()
x0 = data[0]
split_lengths = [int(len(data)*0.6+1), int(len(data)*0.4)]
ds_train, ds_test = random_split(data, split_lengths)
dl_train = DataLoader(ds_train, batch_size=4, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=4, shuffle=True)
raw_sample = torch.from_numpy(x0).float().to(device)
dataload_sample = next(iter(dl_train))
# print(dl_sample)  # Now the shape is 4,1,314
# dl_sample = next(iter(ds))

#raw_sample before dataloader shape is [1,314]
#dataloader_sample after DataLoader is [4,1,314]
#encoder_output sample size is [4,1024,319]
#decoder_output sample size is [4,1,314]


# Model
encoder = auto_encoder.EncoderCNN()
decoder = auto_encoder.DecoderCNN()

#check per one
enc_feedforward = encoder(dataload_sample) # shape should be 4,1024,319, should it be z?
decoded = decoder(enc_feedforward) #shape is 3,1,314 ( back to normal)
vae = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)#(features_encoder, features_decoder, in_size, z_dim)
vae_dp = DataParallel(vae)
z, mu, log_sigma2 = vae.encode(dataload_sample)


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
checkpoint_file_final = f'{checkpoint_file}_final'
if os.path.isfile(f'{checkpoint_file}.pt'):
    os.remove(f'{checkpoint_file}.pt')

def post_epoch_fn(epoch, train_result, test_result, verbose):
    # Plot some samples if this is a verbose epoch
    if verbose:
        samples = vae.sample(n=5)
        fig, _ = plot.tensors_as_images(samples, figsize=(6, 2))
        IPython.display.display(fig)
        plt.close(fig)


if os.path.isfile(f'{checkpoint_file_final}.pt'):
    print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
    checkpoint_file = checkpoint_file_final
else:
    for i in range(len(list_DL_tuples)):
        if os.path.isfile(f'{checkpoint_file_final}.pt'):
            print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
            current_check_point_file = f'{checkpoint_file_final}_' + str(i)
        res = trainer_mse.fit(list_DL_tuples[i][0], list_DL_tuples[i][1],
                              num_epochs=200, early_stopping=20, print_every=10,
                              checkpoints=checkpoint_file,
                              post_epoch_fn=post_epoch_fn)

    for i in range(len(list_DL_tuples)):
        if os.path.isfile(f'{checkpoint_file_final}.pt'):
            print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
            current_check_point_file = f'{checkpoint_file_final}_' + str(i)
        res = trainer_CE.fit(list_DL_tuples[i][0], list_DL_tuples[i][1],
                              num_epochs=200, early_stopping=20, print_every=10,
                              checkpoints=checkpoint_file,
                              post_epoch_fn=post_epoch_fn)


# enc_feedforward = enc(dl_sample) # shape is 4,1024,319
# decoded = dec(enc_feedforward) #shape is 4,1,314
# print(enc_feedforward)



# def test_vae_loss():
#     # Test data
#     N, C, H, W = 10, 3, 64, 64 #TODO: figure out the parameters
#     z_dim = 1 #TODO: figure out the parameters
#     # x = torch.randn(N, C, H, W) * 2 - 1
#     x  = torch.randn(1, *raw_sample.shape)
#     # xr = torch.randn(N, C, H, W) * 2 - 1
#     xr = self.features_decoder(h)
#     z_mu = torch.randn(N, z_dim)
#     z_log_sigma2 = torch.randn(N, z_dim)
#     x_sigma2 = 0.9
#     loss, _, _ = vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2)
#     return loss