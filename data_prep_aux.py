import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import itertools
import aux_data_sensors as aux_loader
import auto_encoder

class AuxDataset(Dataset):
    def __init__(self):
        self.D1_arrays_aux = aux_loader.return_data()

    def __len__(self):
        # return len(self.D1_arrays_energy)
        return len(self.D1_arrays_aux)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.expand_dims(self.D1_arrays_aux[idx], axis=0)
        return sample
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # # sample = np.expand_dims(self.D1_arrays_energy[idx], axis=0)
        # sample = np.expand_dims(self.dataset[idx], axis=0)
        # return sample
        # self.d1_energy[idx * batch_size: (idx + 1) * batch_size]



class data_prep_aux():
    ##for a better explanation please see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self):
        self.aux_ds = AuxDataset()
        split_lengths = [int(len(self.aux_ds) * 0.6), int(len(self.aux_ds) * 0.4) + 1]
        ds_train_aux, ds_test_aux = random_split(self.aux_ds, split_lengths)
        self.data_loader_train_aux = DataLoader(ds_train_aux, batch_size=4, shuffle=True)
        self.data_loader_test_aux = DataLoader(ds_test_aux,batch_size= 1, shuffle=True)

    def getEnergy(self):
        return self.data_loader_train_aux, self.data_loader_test_aux

    def returnDSAux(self):
        dataloader = DataLoader(self.aux_ds,batch_size=1, shuffle=True)
        return dataloader

