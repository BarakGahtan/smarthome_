import os
import numpy as np
import auto_encoder as autoencoder
import thermal_energy_intersect as thermal_energy_loader
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim
import matplotlib.pyplot as plt
import tensorflow as tf
# def hyperparams():
#     hypers = dict(
#         batch_size=0,
#         h_dim=0, z_dim=0, x_sigma2=0,
#         learn_rate=0.0, betas=(0.0, 0.0),
#     )
#     hypers['batch_size'] = 2
#     hypers['h_dim'] = 128
#     hypers['z_dim'] = 64
#     hypers['x_sigma2'] = 0.001
#     hypers['learn_rate'] = 0.001
#     hypers['betas'] = (0.85, 0.999)
#     return hypers
torch.manual_seed(42)

VECTOR_LENGTH = 86400
########################################################################################################################
###################################    thermal and energy autoencoder  #################################################
########################################################################################################################

D1_arrays_energy, D1_arrays_thermal = thermal_energy_loader.return_data()

split_lengths = [int(len(D1_arrays_energy)*0.6), int(len(D1_arrays_energy)*0.4)+1]
ds_train_energy, ds_test_energy = random_split(D1_arrays_energy, split_lengths)
dl_train_energy_dl = DataLoader(ds_train_energy, 4, shuffle=True)
dl_test_energy_dl  = DataLoader(ds_test_energy,  1, shuffle=True)

sample = next(iter(dl_train_energy_dl))


#tensor_list_thermal, tensor_list_thermal_padded,thermal_max, tensor_list_energy,tensor_list_energy_padded,energy_max= thermal_energy_loader.return_data()
#convert the tensors into [max_len,2,1,1]
# for t in tensor_list_energy_padded:
#     # b = torch.ones_like(t)
#     # t1 = torch.cat([t,b],1)
#     t1 = t1.unsqueeze(2)
#     t1 = t1.unsqueeze(3)
#     energy_tensor_list_padded_catted.append(t1)
#
# thermal_tensor_list_padded_catted = []
# for t in tensor_list_thermal_padded:
#     # b = torch.ones_like(t)
#     # t1 = torch.cat([t,b],1)
#     t1 = t1.unsqueeze(2)
#     t1 = t1.unsqueeze(3)
#     thermal_tensor_list_padded_catted.append(t1)
# def load_image(filename):
#   with open(filename) as f:
#     return np.array(f.read())
#
#
# from PIL import Image
# import numpy as np
# import torchvision.transforms as T
# from torchvision.datasets import ImageFolder
# im_size = 64
# tf = T.Compose([
#     # Resize to constant spatial dimensions
#     T.Resize((im_size, im_size)),
#     # PIL.Image -> torch.Tensor
#     T.ToTensor(),
#     # Dynamic range [0,1] -> [-1, 1]
#     T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)),
# ])
# ds_gwb = ImageFolder(os.path.dirname("graphs/graphs energy and thermal days"), tf)
# print("CHEcK")
# im = np.array(Image.open("graphs/graphs energy and thermal days/Thermal, day number 46.png"),np.float)
#
# #x_2 = tf.io.decode_png(x_444, channels=3, dtype=tf.dtypes.uint8, name=None)
#
# # Data
# split_lengths = [int(len(ds_gwb)*0.9), int(len(ds_gwb)*0.1)]
# ds_train, ds_test = random_split(ds_gwb, split_lengths)
# dl_train = DataLoader(ds_train, batch_size, shuffle=True)
# dl_test  = DataLoader(ds_test,  batch_size, shuffle=True)
# # size = thermal_tensor[0].shape
#
# # Model
# encoder = autoencoder.EncoderCNN(in_channels=2, out_channels=out_channels)
#
# decoder = autoencoder.DecoderCNN(in_channels=out_channels, out_channels=2)
#
# #number of inside layers after the second layer is the number of convolution kernels which can be features.