import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import itertools
import aux_data_sensors as aux_loader
import auto_encoder
import thermal_energy_intersect as thermal_energy_loader

def data_wrapper():
    res = []
    aux = aux_loader.return_data()
    thermal = thermal_energy_loader.return_data(0)
    energy = thermal_energy_loader.return_data(1)
    for j in range(len(aux)):
        res.append(aux[j])

    for j in range(len(thermal)):
        res.append(thermal[j])

    for j in range(len(energy)):
        res.append(energy[j])
    return res

class DatasetCombined(Dataset):
    def __init__(self):
        self.data = data_wrapper()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.expand_dims(self.data[idx], axis=0)
        return sample
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # # sample = np.expand_dims(self.D1_arrays_energy[idx], axis=0)
        # sample = np.expand_dims(self.dataset[idx], axis=0)
        # return sample
        # self.d1_energy[idx * batch_size: (idx + 1) * batch_size]


