import os
from scipy.constants import hp
from auto_encoder import vae_loss
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
import IPython.display
import matplotlib.pyplot as plt
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
# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=learn_rate, betas=betas)

# Loss
def loss_fn(x, xr, z_mu, z_log_sigma2):
    return autoencoder.vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2)

# Trainer
trainer = VAETrainer(vae_dp, loss_fn, optimizer, device)

def test_vae_loss():
    # Test data
    N, C, H, W = 10, 3, 64, 64 #TODO: figure out the parameters
    z_dim = 1 #TODO: figure out the parameters
    # x = torch.randn(N, C, H, W) * 2 - 1
    x  = torch.randn(1, *raw_sample.shape)
    # xr = torch.randn(N, C, H, W) * 2 - 1
    xr = self.features_decoder(h)
    z_mu = torch.randn(N, z_dim)
    z_log_sigma2 = torch.randn(N, z_dim)
    x_sigma2 = 0.9
    loss, _, _ = vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2)
    return loss

# test_vae_loss()

checkpoint_file = 'checkpoints/vae'
checkpoint_file_final = f'{checkpoint_file}_final'
if os.path.isfile(f'{checkpoint_file}.pt'):
    os.remove(f'{checkpoint_file}.pt')

# Show model and hypers
print(vae)
print(hp)

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
    res = trainer.fit(dl_train, dl_test,
                      num_epochs=200, early_stopping=20, print_every=10,
                      checkpoints=checkpoint_file,
                      post_epoch_fn=post_epoch_fn)


# enc_feedforward = enc(dl_sample) # shape is 4,1024,319
# decoded = dec(enc_feedforward) #shape is 4,1,314
# print(enc_feedforward)