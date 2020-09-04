from __future__ import print_function, division
import os
import numpy as np
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import auto_encoder as autoencoder
import main_data as dataLoader
from auto_encoder import EncoderCNN, DecoderCNN
from training import PredictorTrainer
from transfer_learning_arch import *
from utils import *
from utils.plot import plot_fit, plot_accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# preparing different datasets
# 0 - aux sensor only
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

# preparing different datasets
# 1 - thermal sensor only
#preparing the labels for training and testings #
data, thermal_labels = dataLoader.DatasetCombined(2)
thermal_list_labels = []
thermal_labels = thermal_labels.squeeze(0)
for i in range(np.shape(thermal_labels)[0]):
    thermal_list_labels.append((thermal_labels[i][0],thermal_labels[i][1]))
ds_train_thermal_labels, ds_test_thermal_labels = random_split(thermal_list_labels, [150,37]) #80\20 train\test
ds_train_lables = dataLoader.Labels(ds_train_thermal_labels)
ds_test_lables = dataLoader.Labels(ds_test_thermal_labels)
dl_train_thermal = DataLoader(ds_train_lables, batch_size=4, shuffle=True)
dl_test_thermal = DataLoader(ds_test_lables, batch_size=4, shuffle=True)

# # 2 - energy sensor only
data, energy_labels_peak_ratio_in_day = dataLoader.DatasetCombined(3)
energy_list_labels = []
energy_labels = energy_labels_peak_ratio_in_day.squeeze(0)
for i in range(np.shape(energy_labels)[0]):
    energy_list_labels.append((energy_labels[i][0],energy_labels[i][1]))
ds_train_energy_labels, ds_test_energy_labels = random_split(energy_list_labels, [99,25]) #80\20 train\test
ds_train_lables_energy = dataLoader.Labels(ds_train_energy_labels)
ds_test_lables_energy = dataLoader.Labels(ds_test_energy_labels)
dl_train_lables_energy = DataLoader(ds_train_lables_energy, batch_size=4, shuffle=True)
dl_test_lables_energy = DataLoader(ds_test_lables_energy, batch_size=4, shuffle=True)

# 5 - energy and thermal
energy_thermal =energy_list_labels + thermal_list_labels
ds_train_energy_thermal_labels, ds_test_energy_thermal_labels = random_split(energy_thermal, [248,63]) #80\20 train\test
ds_train_lables_energy_thermal = dataLoader.Labels(ds_train_energy_thermal_labels)
ds_test_lables_energy_thermal = dataLoader.Labels(ds_train_energy_thermal_labels)
dl_train_lables_energy_thermal = DataLoader(ds_train_lables_energy_thermal, batch_size=4, shuffle=True)
dl_test_lables_energy_thermal = DataLoader(ds_test_lables_energy_thermal, batch_size=4, shuffle=True)



# load models #
loaded_model_1 = torch.load("vae_MSE_latentSpace_1.pt",map_location=torch.device(device))
loaded_model_2 = torch.load("vae_MSE_latentSpace_2.pt",map_location=torch.device(device))
loaded_model_5 = torch.load("vae_MSE_latentSpace_5.pt",map_location=torch.device(device))



#creating the models - vae trained + unfreeze
# init encoder and decoder for loading #
encoder_1 = EncoderCNN()
decoder_1 = DecoderCNN()
predictor_1 = Predictor()
vae_1 = autoencoder.VAE(encoder_1, decoder_1, raw_sample.shape, 1)
vae_dp_1 = DataParallel(vae_1)
# vae_dp_1.load_state_dict(loaded_model_1['model_state'],strict=False)
model_1 = Encoded_Predictor(vae_dp_1,predictor_1)

encoder_2 = EncoderCNN()
decoder_2 = DecoderCNN()
predictor_2 = Predictor()
vae_2 = autoencoder.VAE(encoder_2, decoder_2, raw_sample.shape, 1)
vae_dp_2 = DataParallel(vae_2)
# vae_dp_2.load_state_dict(loaded_model_2['model_state'],strict=False)
model_2 = Encoded_Predictor(vae_dp_2,predictor_2)

encoder_5 = EncoderCNN()
decoder_5 = DecoderCNN()
predictor_5 = Predictor()
vae_5 = autoencoder.VAE(encoder_5, decoder_5, raw_sample.shape, 1)
vae_dp_5 = DataParallel(vae_5)
# vae_dp_5.load_state_dict(loaded_model_5['model_state'],strict=False)
model_5 = Encoded_Predictor(vae_dp_5,predictor_5)

optimizer_1 =  optim.Adam(vae_1.parameters(), lr=learn_rate, betas=betas)
optimizer_2 =  optim.Adam(vae_2.parameters(), lr=learn_rate, betas=betas)
optimizer_5 =  optim.Adam(vae_5.parameters(), lr=learn_rate, betas=betas)

criterion_MSE = {}

trainer_Predictor_1 = PredictorTrainer(model_1, criterion_MSE, optimizer_1, device)
trainer_Predictor_2 = PredictorTrainer(model_2, criterion_MSE, optimizer_2, device)
trainer_Predictor_5 = PredictorTrainer(model_5, criterion_MSE, optimizer_5, device)

trainer_Predictor_list = [trainer_Predictor_1,trainer_Predictor_2,trainer_Predictor_5]

checkpoint_file = 'checkpoints/vae'
checkpoint_file_final = f'{checkpoint_file}_final_predictor_with_load'
if os.path.isfile(f'{checkpoint_file}.pt'):
    os.remove(f'{checkpoint_file}.pt')


if os.path.isfile(f'{checkpoint_file_final}.pt'):
    print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
    checkpoint_file = checkpoint_file_final
else:
    if os.path.isfile(f'{checkpoint_file_final}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        current_check_point_file = f'{checkpoint_file_final}_' + str(i)
    current_check_point_file_1 = f'{checkpoint_file_final}_' + str(1)
    res_1 = trainer_Predictor_1.fit(dl_train_thermal, dl_test_thermal,
                                                num_epochs=200, early_stopping=20, print_every=10,
                                                checkpoints=current_check_point_file_1,
                                                post_epoch_fn=None)
    # plot_fit(res_1)
    plot_accuracy(res_1, "thermal model - VAE - without warm start",accuracy=True)
    # plot_accuracy(res_1, "thermal model - VAE - with warm start",accuracy=False)
    current_check_point_file_2 = f'{checkpoint_file_final}_' + str(2)
    res_2 = trainer_Predictor_2.fit(dl_train_lables_energy, dl_test_thermal,
                                          num_epochs=200, early_stopping=20, print_every=10,
                                          checkpoints=current_check_point_file_2,
                                          post_epoch_fn=None)
    plot_accuracy(res_2, "energy model - VAE - without warm start", accuracy=True)
    # plot_accuracy(res_2, "energy model - VAE - with warm start", accuracy=False)
    current_check_point_file_3 = f'{checkpoint_file_final}_' + str(5)
    res_3 = trainer_Predictor_5.fit(dl_train_lables_energy_thermal,dl_test_thermal,
                                          num_epochs=200, early_stopping=20, print_every=10,
                                          checkpoints=current_check_point_file_3,
                                          post_epoch_fn=None)
    plot_accuracy(res_3, "energy and thermal model - VAE - without warm start", accuracy=True)
    # plot_accuracy(res_3, "energy and thermal model - VAE - with warm start", accuracy=False)
    # fit_results = [ res_1, res_2, res_3]
