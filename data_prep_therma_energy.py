import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import thermal_energy_intersect as thermal_energy_loader
import auto_encoder

class EnergyDataset(Dataset):
    def __init__(self):
        self.D1_arrays_energy, self.D1_arrays_thermal = thermal_energy_loader.return_data()

    def __len__(self):
        return len(self.D1_arrays_energy)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.expand_dims(self.D1_arrays_energy[idx], axis=0)
        return sample
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # # sample = np.expand_dims(self.D1_arrays_energy[idx], axis=0)
        # sample = np.expand_dims(self.dataset[idx], axis=0)
        # return sample
        # self.d1_energy[idx * batch_size: (idx + 1) * batch_size]

class ThermalDataset(Dataset):
    def __init__(self):
        self.D1_arrays_energy, self.D1_arrays_thermal = thermal_energy_loader.return_data()

    def __len__(self):
        # return len(self.D1_arrays_energy)
        return len(self.D1_arrays_thermal)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.expand_dims(self.D1_arrays_thermal[idx], axis=0)
        return sample

class data_prep_Energy():
    ##for a better explanation please see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self):
        self.energy_ds = EnergyDataset()
        split_lengths = [int(len(self.energy_ds) * 0.6), int(len(self.energy_ds) * 0.4) + 1]
        ds_train_energy, ds_test_energy = random_split(self.energy_ds, split_lengths)
        self.data_loader_train_energy = DataLoader(ds_train_energy, batch_size=4, shuffle=True)
        self.data_loader_test_energy = DataLoader(ds_test_energy,batch_size= 1, shuffle=True)

    def getEnergy(self):
        return self.data_loader_train_energy, self.data_loader_test_energy

    def returnDSEnergy(self):
        dataloader = DataLoader(self.energy_ds,batch_size=1, shuffle=True)
        return dataloader

class data_prep_Thermal():
    ##for a better explanation please see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self):
        self.thermal_ds = ThermalDataset()
        split_lengths = [int(len(self.thermal_ds) * 0.6), int(len(self.thermal_ds) * 0.4) + 1]
        ds_train_thermal, ds_test_thermal = random_split(self.thermal_ds, split_lengths)
        self.data_loader_train_thermal = DataLoader(ds_train_thermal, batch_size=4, shuffle=True)
        self.data_loader_test_thermal = DataLoader(ds_test_thermal, batch_size=1, shuffle=True)

    def getEnergy(self):
        return self.data_loader_train_thermal, self.data_loader_test_thermal

    def returnDSThermal(self):
        dataloader = DataLoader(self.thermal_ds,batch_size=1, shuffle=True)
        return dataloader

    # dl_sample = next(iter(dl))
    #
    # print(dl_sample) # Now the shape is 4,1,314
    #
    # enc = auto_encoder.EncoderCNN()
    # dec = auto_encoder.DecoderCNN()
    # enc_feedforward = enc(dl_sample) # shape should be 4,1024,314
    # decoded = dec(enc_feedforward)
    #
    #
    # print(enc_feedforward)
    # print(decoded)
