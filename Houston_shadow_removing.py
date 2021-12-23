
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.io as sio # package of loading .mat 
import numpy as np
from datasets import *
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from model_Houston import GeneratorResNet
import matplotlib.pyplot as plt
import h5py

file_dir = './data_Houston.mat'
model_path = './saved_models/Houston_removal_shadow/G_AB_2990.pth' 

data = sio.loadmat(file_dir)
#data = h5py.File(file_dir,'r')
pavia_data_final = data['data_Houston']
print(pavia_data_final.shape)
print(type(pavia_data_final))
pavia_data_final = pavia_data_final.astype(np.float32)
pavia_data_final = pavia_data_final.reshape(349*1905,144)
data_tensor = torch.from_numpy(pavia_data_final)


model = GeneratorResNet()
model.load_state_dict(torch.load(model_path))

device="cuda:0"                
model=model.to(device)
model.eval()

data_tensor = data_tensor.to(device)
output = model(data_tensor)
output = output.to('cpu').detach().numpy()
output = output.reshape(349,1905,144)
print(output.shape)
sio.savemat('./data_Houston_reconstruction.mat',{'data_Houston_reconstruction':output})
