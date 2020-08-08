import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import itertools
import aux_data_sensors as aux_loader
import auto_encoder
import thermal_energy_intersect as thermal_energy_loader
import tensorflow as tf
def data_wrapper():
    res = []
    aux = aux_loader.return_data()
    thermal = thermal_energy_loader.return_data(0)
    energy = thermal_energy_loader.return_data(1)
    for j in range(len(aux)):
        res.append(tf.convert_to_tensor(aux[j],dtype=tf.float32)) #before it was only insert without conversion
    for j in range(len(thermal)):
        res.append(tf.convert_to_tensor(thermal[j],dtype=tf.float32))  #before it was only insert without conversion
    for j in range(len(energy)):
        res.append(tf.convert_to_tensor(energy[j],dtype=tf.float32))  #before it was only insert without conversion
    max = tf.reduce_max(tf.stack(res))
    min = tf.reduce_min(tf.stack(res))
    normalized_results = []
    for t in res:
        normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x-min) / (max-min)), t))
    return normalized_results

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



