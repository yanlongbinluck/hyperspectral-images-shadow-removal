
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import scipy.io as sio # package of loading .mat 
import numpy as np
from datasets import *
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from model_fru_grass import GeneratorResNet
import matplotlib.pyplot as plt
import h5py

data = 'fru_shadow_resize.mat' # TODO
model_dir = 'pair_fru'
file_dir = './' + data
model_path = './saved_models/'+ model_dir +'_removal_shadow/G_AB_990.pth'

data = h5py.File(file_dir,'r')
pavia_data_final = data['out'].value
pavia_data_final = pavia_data_final.transpose((2,1,0))
pavia_data_final = pavia_data_final.astype(np.float32)
#pavia_data_final = pavia_data_final[0:400,:,:] # total: [1702,1392,256] split to [1000,1392,256] and [702,1392,256]
print(pavia_data_final.shape)
print(type(pavia_data_final))
m,n,l = pavia_data_final.shape
pavia_data_final = pavia_data_final.reshape(m*n,256)

data_tensor = torch.from_numpy(pavia_data_final)


model = GeneratorResNet()
model.load_state_dict(torch.load(model_path))

device="cuda:0"                
model=model.to(device)
model.eval()

data_tensor = data_tensor.to(device)
output = model(data_tensor)
output = output.to('cpu').detach().numpy()
output = output.reshape(m,n,l)
print(output.shape)
sio.savemat('./data_'+'_reconstruction.mat',{'data_'+'_reconstruction':output})
