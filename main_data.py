import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset
import aux_data_sensors as aux_loader
import thermal_energy_intersect as thermal_energy_loader


def data_wrapper(flag):
    if flag == 1:#only aux sensors.
        res = []
        aux = aux_loader.return_data()
        for j in range(len(aux)):
            res.append(tf.convert_to_tensor(aux[j],dtype=tf.float32)) #before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x - min) / (max - min)), t))
        return normalized_results
    if flag == 2:#only thermal sensors.
        res = []
        thermal,thermal_labels_peak_ratio_in_day = thermal_energy_loader.return_data(0)
        for j in range(len(thermal)):
            res.append(tf.convert_to_tensor(thermal[j], dtype=tf.float32))  # before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x - min) / (max - min)), t))
        return normalized_results,thermal_labels_peak_ratio_in_day
    if flag == 3:#only energy sensors.
        res = []
        energy,energy_labels_peak_ratio_in_day = thermal_energy_loader.return_data(1)
        for j in range(len(energy)):
            res.append(tf.convert_to_tensor(energy[j],dtype=tf.float32))  #before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x-min) / (max-min)), t))
        return normalized_results,energy_labels_peak_ratio_in_day
    if flag == 4: #only thermal and aux sensors.
        res = []
        aux = aux_loader.return_data()
        thermal,thermal_labels_peak_ratio_in_day = thermal_energy_loader.return_data(0)
        for j in range(len(aux)):
            res.append(tf.convert_to_tensor(aux[j], dtype=tf.float32))  # before it was only insert without conversion
        for j in range(len(thermal)):
            res.append(
                tf.convert_to_tensor(thermal[j], dtype=tf.float32))  # before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x - min) / (max - min)), t))
        return normalized_results,thermal_labels_peak_ratio_in_day
    if flag == 5: #only energy and aux sensors.
        res = []
        aux = aux_loader.return_data()
        energy, energy_labels_peak_ratio_in_day = thermal_energy_loader.return_data(1)
        for j in range(len(aux)):
            res.append(tf.convert_to_tensor(aux[j], dtype=tf.float32))  # before it was only insert without conversion
        for j in range(len(energy)):
            res.append(
                tf.convert_to_tensor(energy[j], dtype=tf.float32))  # before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x - min) / (max - min)), t))
        return normalized_results,energy_labels_peak_ratio_in_day
    if flag == 6: #only energy and thermal sensors.
        res = []
        thermal,thermal_labels_peak_ratio_in_day = thermal_energy_loader.return_data(0)
        energy, energy_labels_peak_ratio_in_day = thermal_energy_loader.return_data(1)
        for j in range(len(thermal)):
            res.append(
                tf.convert_to_tensor(thermal[j], dtype=tf.float32))  # before it was only insert without conversion
        for j in range(len(energy)):
            res.append(
                tf.convert_to_tensor(energy[j], dtype=tf.float32))  # before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x - min) / (max - min)), t))
        return normalized_results,thermal_labels_peak_ratio_in_day,energy_labels_peak_ratio_in_day
    if flag == 7:# aux, energy and thermal sensors.
        res = []
        aux = aux_loader.return_data()
        thermal,thermal_labels_peak_ratio_in_day = thermal_energy_loader.return_data(0)
        energy,energy_labels_peak_ratio_in_day = thermal_energy_loader.return_data(1)
        for j in range(len(aux)):
            res.append(tf.convert_to_tensor(aux[j], dtype=tf.float32))  # before it was only insert without conversion
        for j in range(len(thermal)):
            res.append(
                tf.convert_to_tensor(thermal[j], dtype=tf.float32))  # before it was only insert without conversion
        for j in range(len(energy)):
            res.append(
                tf.convert_to_tensor(energy[j], dtype=tf.float32))  # before it was only insert without conversion
        max = tf.reduce_max(tf.stack(res))
        min = tf.reduce_min(tf.stack(res))
        normalized_results = []
        for t in res:
            normalized_results.append(tf.ragged.map_flat_values(lambda x: abs((x - min) / (max - min)), t))
        return normalized_results,thermal_labels_peak_ratio_in_day,energy_labels_peak_ratio_in_day

class DatasetCombined(Dataset):
    def __init__(self, num):
        self.data = data_wrapper(num)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.expand_dims(self.data[idx], axis=0)
        return sample

    def get_labels(self):
        return data_wrapper(2)

class Labels(torch.utils.data.Dataset):
  def __init__(self, data):
        self.data = data

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        sample = np.expand_dims(self.data[index][0], axis=0)
        return {
            'array': sample,
            'label' : self.data[index][1]
            }