from __future__ import print_function, division
import copy
from datetime import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# import main_data as dataLoader
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
    def __init__(self, in_channels=1024, out_channels = 1):
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_channels=in_channels, out_channels=64))
        modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Linear(64, 4))
        modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)

# # preparing different datasets
# # 1 - aux sensor only
# data_aux = dataLoader.DatasetCombined(1)
# split_lengths_aux = [int(len(data_aux)*1), int(len(data_aux)*0)]
# ds_train_aux, ds_test_aux = random_split(data_aux, split_lengths_aux)
# dl_train_0 = DataLoader(ds_train_aux, batch_size=4, shuffle=True)
# dl_test_0 = DataLoader(ds_test_aux, batch_size=4, shuffle=True)
#
# # 2 - thermal sensor only
# data_thermal,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(2)
# split_lengths_thermal = [int(len(data_thermal)*1), int(len(data_thermal)*0)]
# ds_train_thermal, ds_test_thermal = random_split(data_thermal, split_lengths_thermal)
# dl_train_1 = DataLoader(ds_train_thermal, batch_size=4, shuffle=True)
# dl_test_1 = DataLoader(ds_test_thermal, batch_size=4, shuffle=True)
#
# # 3 - energy sensor only
# data_energy,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(3)
# split_lengths_energy = [int(len(data_energy)*1), int(len(data_energy)*0)]
# ds_train_energy, ds_test_energy = random_split(data_energy, split_lengths_energy)
# dl_train_2 = DataLoader(ds_train_energy, batch_size=4, shuffle=True)
# dl_test_2 = DataLoader(ds_test_energy, batch_size=4, shuffle=True)
#
# # 4 - thermal and aux
# data_thermal_aux,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(4)
# split_lengths_thermal_aux = [int(len(data_thermal_aux)*1), int(len(data_thermal_aux)*0)]
# ds_train_thermal_aux, ds_test_thermal_aux = random_split(data_thermal_aux, split_lengths_thermal_aux)
# dl_train_3 = DataLoader(ds_train_thermal_aux, batch_size=4, shuffle=True)
# dl_test_3 = DataLoader(ds_test_thermal_aux, batch_size=4, shuffle=True)
#
# # 5 - energy and aux
# data_energy_aux,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(5)
# split_lengths_energy_aux = [int(len(data_energy_aux)*1), int(len(data_energy_aux)*0)]
# ds_train_energy_aux, ds_test_energy_aux = random_split(data_energy_aux, split_lengths_energy_aux)
# dl_train_4 = DataLoader(ds_train_energy_aux, batch_size=4, shuffle=True)
# dl_test_4 = DataLoader(ds_test_energy_aux, batch_size=4, shuffle=True)
#
# # 6 - energy and thermal
# data_energy_thermal,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(6)
# split_lengths_energy_thermal = [int(len(data_energy_thermal)*1), int(len(data_energy_thermal)*0)]
# ds_train_energy_thermal, ds_test_energy_thermal = random_split(data_energy_thermal, split_lengths_energy_thermal)
# dl_train_5 = DataLoader(ds_train_energy_thermal, batch_size=4, shuffle=True)
# dl_test_5 = DataLoader(ds_test_energy_thermal, batch_size=4, shuffle=True)
#
# # 7 - energy, thermal and aux
# data_energy_thermal_aux,thermal_labels_peak_ratio_in_day = dataLoader.DatasetCombined(7)
# split_lengths_energy_thermal_aux = [int(len(data_energy_thermal_aux)*1), int(len(data_energy_thermal_aux)*0)]
# ds_train_energy_thermal_aux, ds_test_energy_thermal_aux = random_split(data_energy_thermal_aux, split_lengths_energy_thermal_aux)
# dl_train_6 = DataLoader(ds_train_energy_thermal_aux, batch_size=4, shuffle=True)
# dl_test_6 = DataLoader(ds_test_energy_thermal_aux, batch_size=4, shuffle=True)

loaded_model_0 = torch.load("vae_final_0.pt",map_location=torch.device('cpu'))
loaded_model_1 = torch.load("vae_final_1.pt",map_location=torch.device('cpu'))
loaded_model_2 = torch.load("vae_final_2.pt",map_location=torch.device('cpu'))
loaded_model_3 = torch.load("vae_final_3.pt",map_location=torch.device('cpu'))
loaded_model_4 = torch.load("vae_final_4.pt",map_location=torch.device('cpu'))
loaded_model_5 = torch.load("vae_final_5.pt",map_location=torch.device('cpu'))
loaded_model_6 = torch.load("vae_final_6.pt",map_location=torch.device('cpu'))

def activate_layers(model):
    for current_layer in model.items():
        x=6

activate_layers(loaded_model_0)


data, thermal_labels = dataLoader.DatasetCombined(2)
split_lengths_thermal_labels = [int(len(thermal_labels)*0.8), int(len(thermal_labels)*0.2)]
ds_train_thermal_labels, ds_test_thermal_labels = random_split(thermal_labels.squeeze(0), [150,37]) #80\20 train\test
ds_train_lables = Labels(ds_train_thermal_labels)
ds_test_lables = Labels(ds_test_thermal_labels)
dl_train_lables = DataLoader(ds_train_lables, batch_size=4, shuffle=True)
dl_test_lables = DataLoader(ds_test_lables, batch_size=4, shuffle=True)








def train_model(model, criterion, optimizer, scheduler,  dl_train, dl_test, num_epochs=25,):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

dl_train_list = [dl_train_0, dl_train_1, dl_train_2, dl_train_3, dl_train_4, dl_train_5, dl_train_6]
dl_test_0 = DataLoader(ds_test_aux, batch_size=4, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer_0 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_1 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_2 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_3 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_4 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_5 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)
optimizer_6 = optim.SGD(loaded_model_0.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler_0 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_1 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_2 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_3 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_4 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_5 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)
exp_lr_scheduler_6 = lr_scheduler.StepLR(optimizer_0, step_size=7, gamma=0.1)

criterion_CE = nn.CrossEntropyLoss()
criterion_MSE = nn.MSELoss()

model_ft_0_CE = train_model(loaded_model_0, criterion_CE, optimizer_0, exp_lr_scheduler_0,num_epochs=25)
model_ft_1_CE = train_model(loaded_model_1, criterion_CE, optimizer_1, exp_lr_scheduler_1,num_epochs=25)
model_ft_2_CE = train_model(loaded_model_2, criterion_CE, optimizer_2, exp_lr_scheduler_2,num_epochs=25)
model_ft_3_CE = train_model(loaded_model_3, criterion_CE, optimizer_3, exp_lr_scheduler_3,num_epochs=25)
model_ft_4_CE = train_model(loaded_model_4, criterion_CE, optimizer_4, exp_lr_scheduler_4,num_epochs=25)
model_ft_5_CE = train_model(loaded_model_5, criterion_CE, optimizer_5, exp_lr_scheduler_5,num_epochs=25)
model_ft_6_CE = train_model(loaded_model_6, criterion_CE, optimizer_6, exp_lr_scheduler_6,num_epochs=25)

model_ft_0_MSE= train_model(loaded_model_0, criterion_MSE, optimizer_0, exp_lr_scheduler_0,num_epochs=25)
model_ft_1_MSE = train_model(loaded_model_1, criterion_MSE, optimizer_1, exp_lr_scheduler_1,num_epochs=25)
model_ft_2_MSE = train_model(loaded_model_2, criterion_MSE, optimizer_2, exp_lr_scheduler_2,num_epochs=25)
model_ft_3_MSE = train_model(loaded_model_3, criterion_MSE, optimizer_3, exp_lr_scheduler_3,num_epochs=25)
model_ft_4_MSE = train_model(loaded_model_4, criterion_MSE, optimizer_4, exp_lr_scheduler_4,num_epochs=25)
model_ft_5_MSE = train_model(loaded_model_5, criterion_MSE, optimizer_5, exp_lr_scheduler_5,num_epochs=25)
model_ft_6_MSE = train_model(loaded_model_6, criterion_MSE, optimizer_6, exp_lr_scheduler_6,num_epochs=25)