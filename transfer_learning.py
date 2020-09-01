from __future__ import print_function, division
import copy
import os
from datetime import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import auto_encoder
import auto_encoder as autoencoder
from auto_encoder import EncoderCNN, DecoderCNN
import main_data as dataLoader
from training import VAETrainer, PredictorTrainer
from torch.nn import DataParallel

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
hp = vae_hyperparams()
learn_rate = hp['learn_rate']

class Labels(torch.utils.data.Dataset):
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        return data[index][0], data[index][1]


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(1306624, 64)
        self.fc2 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(64, 16)
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
        # x = torch.cat((x1, x2), dim=1)
        return x2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # preparing different datasets
# 1 - aux sensor only
data_aux = dataLoader.DatasetCombined(1)
split_lengths_aux = [int(len(data_aux)*1), int(len(data_aux)*0)]
ds_train_aux, ds_test_aux = random_split(data_aux, split_lengths_aux)
dl_train_0 = DataLoader(ds_train_aux, batch_size=4, shuffle=True)
x0 = data_aux[0]
raw_sample = torch.from_numpy(x0).float().to(device)
dataload_sample = next(iter(dl_train_0))

# 2 - thermal sensor only
data_thermal,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(2)
split_lengths_thermal = [int(len(data_thermal)*1), int(len(data_thermal)*0)]
ds_train_thermal, ds_test_thermal = random_split(data_thermal, split_lengths_thermal)
dl_train_1 = DataLoader(ds_train_thermal, batch_size=4, shuffle=True)

# 3 - energy sensor only
data_energy,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(3)
split_lengths_energy = [int(len(data_energy)*1), int(len(data_energy)*0)]
ds_train_energy, ds_test_energy = random_split(data_energy, split_lengths_energy)
dl_train_2 = DataLoader(ds_train_energy, batch_size=4, shuffle=True)

# 4 - thermal and aux
data_thermal_aux,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(4)
split_lengths_thermal_aux = [int(len(data_thermal_aux)*1), int(len(data_thermal_aux)*0)]
ds_train_thermal_aux, ds_test_thermal_aux = random_split(data_thermal_aux, split_lengths_thermal_aux)
dl_train_3 = DataLoader(ds_train_thermal_aux, batch_size=4, shuffle=True)

# 5 - energy and aux
data_energy_aux,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(5)
split_lengths_energy_aux = [int(len(data_energy_aux)*1), int(len(data_energy_aux)*0)]
ds_train_energy_aux, ds_test_energy_aux = random_split(data_energy_aux, split_lengths_energy_aux)
dl_train_4 = DataLoader(ds_train_energy_aux, batch_size=4, shuffle=True)

# 6 - energy and thermal
data_energy_thermal,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(6)
split_lengths_energy_thermal = [int(len(data_energy_thermal)*1), int(len(data_energy_thermal)*0)]
ds_train_energy_thermal, ds_test_energy_thermal = random_split(data_energy_thermal, split_lengths_energy_thermal)
dl_train_5 = DataLoader(ds_train_energy_thermal, batch_size=4, shuffle=True)

# 7 - energy, thermal and aux
data_energy_thermal_aux,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(7)
split_lengths_energy_thermal_aux = [int(len(data_energy_thermal_aux)*1), int(len(data_energy_thermal_aux)*0)]
ds_train_energy_thermal_aux, ds_test_energy_thermal_aux = random_split(data_energy_thermal_aux, split_lengths_energy_thermal_aux)
dl_train_6 = DataLoader(ds_train_energy_thermal_aux, batch_size=4, shuffle=True)
# dl_train_list = [dl_train_0, dl_train_1, dl_train_2, dl_train_3, dl_train_4, dl_train_5, dl_train_6]

# load models #
loaded_model_0 = torch.load("vae_final_0.pt",map_location=torch.device('cpu'))
loaded_model_1 = torch.load("vae_final_1.pt",map_location=torch.device('cpu'))
loaded_model_2 = torch.load("vae_final_2.pt",map_location=torch.device('cpu'))
loaded_model_3 = torch.load("vae_final_3.pt",map_location=torch.device('cpu'))
loaded_model_4 = torch.load("vae_final_4.pt",map_location=torch.device('cpu'))
loaded_model_5 = torch.load("vae_final_5.pt",map_location=torch.device('cpu'))
loaded_model_6 = torch.load("vae_final_6.pt",map_location=torch.device('cpu'))

# init encoder and decoder for loading #
encoder = EncoderCNN()
decoder = DecoderCNN()
#creating the models
vae_0 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)## enc_feedforward = encoder(dataload_sample) # shape should be 4,1024,319
vae_dp_0 = DataParallel(vae_0)
vae_dp_0.load_state_dict(loaded_model_0['model_state'])

vae_1 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_1 = DataParallel(vae_1)
vae_dp_1.load_state_dict(loaded_model_1['model_state'])

vae_2 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_2 = DataParallel(vae_2)
vae_dp_2.load_state_dict(loaded_model_2['model_state'])

vae_3 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_3 = DataParallel(vae_3)
vae_dp_3.load_state_dict(loaded_model_3['model_state'])

vae_4 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_4 = DataParallel(vae_4)
vae_dp_4.load_state_dict(loaded_model_4['model_state'])

vae_5 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_5 = DataParallel(vae_5)
vae_dp_5.load_state_dict(loaded_model_5['model_state'])

vae_6 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_6 = DataParallel(vae_6)
vae_dp_6.load_state_dict(loaded_model_6['model_state'])


z = encoder(dataload_sample)
z_flatten = torch.flatten(z)

predictor_0 = Predictor()
model_0 = Encoded_Predictor(vae_dp_0,predictor_0)
model_1 = Encoded_Predictor(vae_dp_1,predictor_0)
model_2 = Encoded_Predictor(vae_dp_2,predictor_0)
model_3 = Encoded_Predictor(vae_dp_3,predictor_0)
model_4 = Encoded_Predictor(vae_dp_4,predictor_0)
model_5 = Encoded_Predictor(vae_dp_5,predictor_0)
model_6 = Encoded_Predictor(vae_dp_6,predictor_0)

model_0.forward(dataload_sample)
model_1.forward(dataload_sample)
model_2.forward(dataload_sample)
model_3.forward(dataload_sample)
model_4.forward(dataload_sample)
model_5.forward(dataload_sample)
model_6.forward(dataload_sample)


#preparing the labels for training and testings #
data, thermal_labels = dataLoader.DatasetCombined(2)
ds_train_thermal_labels, ds_test_thermal_labels = random_split(thermal_labels.squeeze(0), [150,37]) #80\20 train\test
ds_train_lables = Labels(ds_train_thermal_labels)
ds_test_lables = Labels(ds_test_thermal_labels)
dl_train_lables = DataLoader(ds_train_lables, batch_size=4, shuffle=True)
dl_test_lables = DataLoader(ds_test_lables, batch_size=4, shuffle=True)

dl_test_0 = DataLoader(ds_test_aux, batch_size=4, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer_0 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_1 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_2 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_3 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_4 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_5 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_6 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_list = [optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4, optimizer_5, optimizer_6]

exp_lr_scheduler_0 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_1 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_2 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_3 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_4 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_5 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_6 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)

criterion_CE = nn.CrossEntropyLoss()
criterion_MSE = nn.MSELoss()

trainer_Predictor_0 = PredictorTrainer(Predictor, criterion_MSE, optimizer_0, device)
trainer_Predictor_1 = PredictorTrainer(Predictor, criterion_MSE, optimizer_1, device)
trainer_Predictor_2 = PredictorTrainer(Predictor, criterion_MSE, optimizer_2, device)
trainer_Predictor_3 = PredictorTrainer(Predictor, criterion_MSE, optimizer_3, device)
trainer_Predictor_4 = PredictorTrainer(Predictor, criterion_MSE, optimizer_4, device)
trainer_Predictor_5 = PredictorTrainer(Predictor, criterion_MSE, optimizer_5, device)
trainer_Predictor_6 = PredictorTrainer(Predictor, criterion_MSE, optimizer_6, device)


trainer_Predictor_list = [trainer_Predictor_0, trainer_Predictor_1,trainer_Predictor_2,trainer_Predictor_3,trainer_Predictor_4,trainer_Predictor_5,trainer_Predictor_6]
# test_vae_loss()
checkpoint_file = 'checkpoints/vae'
checkpoint_file_final = f'{checkpoint_file}_final_predictor'
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
    fit_results = []
    for i in range(len(dl_train_list)):
        if os.path.isfile(f'{checkpoint_file_final}.pt'):
            print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
            current_check_point_file = f'{checkpoint_file_final}_' + str(i)
        current_check_point_file = f'{checkpoint_file_final}_' + str(i)
        res = trainer_Predictor_list[i].fit(dl_train_list[i], dl_train_lables,
                              num_epochs=200, early_stopping=20, print_every=10,
                              checkpoints=current_check_point_file,
                              post_epoch_fn=None)
        fit_results.append(res)

# model_ft_0_CE = train_model(loaded_model_0, criterion_CE, optimizer_0, exp_lr_scheduler_0,num_epochs=25)
# model_ft_1_CE = train_model(loaded_model_1, criterion_CE, optimizer_1, exp_lr_scheduler_1,num_epochs=25)
# model_ft_2_CE = train_model(loaded_model_2, criterion_CE, optimizer_2, exp_lr_scheduler_2,num_epochs=25)
# model_ft_3_CE = train_model(loaded_model_3, criterion_CE, optimizer_3, exp_lr_scheduler_3,num_epochs=25)
# model_ft_4_CE = train_model(loaded_model_4, criterion_CE, optimizer_4, exp_lr_scheduler_4,num_epochs=25)
# model_ft_5_CE = train_model(loaded_model_5, criterion_CE, optimizer_5, exp_lr_scheduler_5,num_epochs=25)
# model_ft_6_CE = train_model(loaded_model_6, criterion_CE, optimizer_6, exp_lr_scheduler_6,num_epochs=25)
#
# model_ft_0_MSE= train_model(loaded_model_0, criterion_MSE, optimizer_0, exp_lr_scheduler_0,num_epochs=25)
# model_ft_1_MSE = train_model(loaded_model_1, criterion_MSE, optimizer_1, exp_lr_scheduler_1,num_epochs=25)
# model_ft_2_MSE = train_model(loaded_model_2, criterion_MSE, optimizer_2, exp_lr_scheduler_2,num_epochs=25)
# model_ft_3_MSE = train_model(loaded_model_3, criterion_MSE, optimizer_3, exp_lr_scheduler_3,num_epochs=25)
# model_ft_4_MSE = train_model(loaded_model_4, criterion_MSE, optimizer_4, exp_lr_scheduler_4,num_epochs=25)
# model_ft_5_MSE = train_model(loaded_model_5, criterion_MSE, optimizer_5, exp_lr_scheduler_5,num_epochs=25)
# model_ft_6_MSE = train_model(loaded_model_6, criterion_MSE, optimizer_6, exp_lr_scheduler_6,num_epochs=25)