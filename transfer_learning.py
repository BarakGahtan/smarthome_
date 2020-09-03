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
from cs236781 import plot
from training import VAETrainer, PredictorTrainer
from torch.nn import DataParallel
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

# load models #
loaded_model_0 = torch.load("vae_final_0.pt",map_location=torch.device('cpu'))
loaded_model_1 = torch.load("vae_final_1.pt",map_location=torch.device('cpu'))
loaded_model_2 = torch.load("vae_final_2.pt",map_location=torch.device('cpu'))
loaded_model_3 = torch.load("vae_final_3.pt",map_location=torch.device('cpu'))
loaded_model_4 = torch.load("vae_final_4.pt",map_location=torch.device('cpu'))
loaded_model_5 = torch.load("vae_final_5.pt",map_location=torch.device('cpu'))
loaded_model_6 = torch.load("vae_final_6.pt",map_location=torch.device('cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

class Labels(torch.utils.data.Dataset):
  def __init__(self, data):
        # max = 0
        # for i in range(len(data)):
        #     current_numpy = data[i][0]
        #     current_shape = np.shape(current_numpy)[0]
        #     if current_shape > max:
        #         max = current_shape
        # new_list = []
        # shape = (max,2)
        # for i in range(len(data)):
        #     res = np.zeros(shape)
        #     res[:data[i][0].shape[0],:data[i][0].shape[1]] = data[i][0]
        #     new_list.append( (res,data[i][1]))
        self.data = data

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        # print(index)
        sample = np.expand_dims(self.data[index][0], axis=0)
        return {
            'array': sample,
            'label' : self.data[index][1]
            }


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

##########
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
#preparing the labels for training and testings #
data, thermal_labels = dataLoader.DatasetCombined(2)
thermal_list_labels = []
thermal_labels = thermal_labels.squeeze(0)
for i in range(np.shape(thermal_labels)[0]):
    thermal_list_labels.append((thermal_labels[i][0],thermal_labels[i][1]))

ds_train_thermal_labels, ds_test_thermal_labels = random_split(thermal_list_labels, [150,37]) #80\20 train\test
ds_train_lables = Labels(ds_train_thermal_labels)
ds_test_lables = Labels(ds_test_thermal_labels)
dl_train_thermal = DataLoader(ds_train_lables, batch_size=4, shuffle=True)
dl_test_thermal = DataLoader(ds_test_lables, batch_size=4, shuffle=True)
#
# # 3 - energy sensor only
data, energy_labels_peak_ratio_in_day = dataLoader.DatasetCombined(3)
energy_list_labels = []
# energy_labels_peak_ratio_in_day = energy_labels_peak_ratio_in_day.squeeze(0)
for i in range(np.shape(energy_labels_peak_ratio_in_day)[0]):
    energy_list_labels.append((energy_labels_peak_ratio_in_day[i][0],energy_labels_peak_ratio_in_day[i][1]))


ds_train_energy_labels, ds_test_energy_labels = random_split(energy_list_labels, [99,25]) #80\20 train\test
ds_train_lables_energy = Labels(ds_train_energy_labels)
ds_test_lables_energy = Labels(ds_test_energy_labels)
dl_train_lables_energy = DataLoader(ds_train_lables_energy, batch_size=4, shuffle=True)
dl_test_lables_energy = DataLoader(ds_test_lables_energy, batch_size=4, shuffle=True)
# 6 - energy and thermal
energy_thermal =energy_list_labels + thermal_list_labels
ds_train_energy_thermal_labels, ds_test_energy_thermal_labels = random_split(energy_thermal, [248,63]) #80\20 train\test
ds_train_lables_energy_thermal = Labels(ds_train_energy_thermal_labels)
ds_test_lables_energy_thermal = Labels(ds_train_energy_thermal_labels)
dl_train_lables_energy_thermal = DataLoader(ds_train_lables_energy_thermal, batch_size=4, shuffle=True)
dl_test_lables_energy_thermal = DataLoader(ds_test_lables_energy_thermal, batch_size=4, shuffle=True)

# dl_train_list = [[dl_train_thermal,dl_test_thermal], [dl_train_lables_energy,dl_test_lables_energy],[dl_train_lables_energy_thermal,dl_test_lables_energy_thermal]]

# load models #

loaded_model_1 = torch.load("vae_final_1.pt",map_location=torch.device(device))
loaded_model_2 = torch.load("vae_final_2.pt",map_location=torch.device(device))
loaded_model_5 = torch.load("vae_final_5.pt",map_location=torch.device(device))

# init encoder and decoder for loading #
encoder = EncoderCNN()
decoder = DecoderCNN()
predictor_0 = Predictor()
#creating the models - vae trained + unfreeze
vae_1 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_1 = DataParallel(vae_1)
vae_dp_1.load_state_dict(loaded_model_1['model_state'])
model_1 = Encoded_Predictor(vae_dp_1,predictor_0)

vae_2 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_2 = DataParallel(vae_2)
vae_dp_2.load_state_dict(loaded_model_2['model_state'])
model_2 = Encoded_Predictor(vae_dp_2,predictor_0)

vae_5 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_5 = DataParallel(vae_5)
vae_dp_5.load_state_dict(loaded_model_5['model_state'])
model_5 = Encoded_Predictor(vae_dp_5,predictor_0)

#creating the models - vae trained + freezed
vae_1 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_1 = DataParallel(vae_1)
vae_dp_1.load_state_dict(loaded_model_1['model_state'])

model_1 = Encoded_Predictor(vae_dp_1,predictor_0)

vae_2 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_2 = DataParallel(vae_2)
vae_dp_2.load_state_dict(loaded_model_2['model_state'])
model_2 = Encoded_Predictor(vae_dp_2,predictor_0)

vae_5 = autoencoder.VAE(encoder, decoder, raw_sample.shape, 1)
vae_dp_5 = DataParallel(vae_5)
vae_dp_5.load_state_dict(loaded_model_5['model_state'])
model_5 = Encoded_Predictor(vae_dp_5,predictor_0)

# z = encoder(dataload_sample)
# z_flatten = torch.flatten(z)







optimizer_1 = optimizer = optim.Adam(vae_1.parameters(), lr=learn_rate, betas=betas)
optimizer_2 = optimizer = optim.Adam(vae_2.parameters(), lr=learn_rate, betas=betas)
optimizer_5 = optimizer = optim.Adam(vae_5.parameters(), lr=learn_rate, betas=betas)

criterion_CE = nn.CrossEntropyLoss()
criterion_MSE = {}

trainer_Predictor_1 = PredictorTrainer(model_1, criterion_MSE, optimizer_1, device)
trainer_Predictor_2 = PredictorTrainer(model_2, criterion_MSE, optimizer_2, device)
trainer_Predictor_5 = PredictorTrainer(model_5, criterion_MSE, optimizer_5, device)

trainer_Predictor_1_CE = PredictorTrainer(model_1, criterion_CE, optimizer_1, device)
trainer_Predictor_2_CE = PredictorTrainer(model_2, criterion_CE, optimizer_2, device)
trainer_Predictor_5_CE = PredictorTrainer(model_5, criterion_CE, optimizer_5, device)

trainer_Predictor_list = [trainer_Predictor_1,trainer_Predictor_2,trainer_Predictor_5]
# trainer_Predictor_list = [trainer_Predictor_1_CE,trainer_Predictor_2_CE,trainer_Predictor_5_CE]
# test_vae_loss()
checkpoint_file = 'checkpoints/vae'
checkpoint_file_final = f'{checkpoint_file}_final_predictor'
if os.path.isfile(f'{checkpoint_file}.pt'):
    os.remove(f'{checkpoint_file}.pt')

def plot_accuracy(results, model_title):
    plt.plot(results.train_acc)
    plt.plot(results.test_acc)
    plt.title(model_title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim((70,100))
    plt.legend(['train', 'test'], loc='lower right')
    plt.text(20, 80., str("average train: "+ str(np.mean(results.train_acc))), family="monospace")
    plt.text(20, 75., str("average test: " + str(np.mean(results.test_acc))), family="monospace")
    # plt.show()
    plt.savefig(str(model_title+".png"))
    plt.close()

print("GOT TO TRAINING PHASE")
if os.path.isfile(f'{checkpoint_file_final}.pt'):
    print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
    checkpoint_file = checkpoint_file_final
else:
    if os.path.isfile(f'{checkpoint_file_final}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        current_check_point_file = f'{checkpoint_file_final}_' + str(i)
    current_check_point_file = f'{checkpoint_file_final}_' + str(i)
    res_1 = trainer_Predictor_1.fit(dl_train_thermal, dl_test_thermal,
                                                num_epochs=200, early_stopping=20, print_every=10,
                                                checkpoints=current_check_point_file,
                                                post_epoch_fn=None)
    plot_accuracy(res_1, "thermal model unfreezed")
    res_2 = trainer_Predictor_2.fit(dl_train_lables_energy, dl_test_thermal,
                                          num_epochs=200, early_stopping=20, print_every=10,
                                          checkpoints=current_check_point_file,
                                          post_epoch_fn=None)
    plot_accuracy(res_2, "energy model unfreezed")
    res_3 = trainer_Predictor_5.fit(dl_train_lables_energy_thermal,dl_test_thermal,
                                          num_epochs=200, early_stopping=20, print_every=10,
                                          checkpoints=current_check_point_file,
                                          post_epoch_fn=None)
    plot_accuracy(res_3, "energy and thermal model unfreezed")
    # fit_results = [ res_1, res_2, res_3]
