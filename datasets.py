
import glob
import random
import os
import scipy.io as sio
import torch
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, dataset_name=None, unaligned=True, mode='train'):
        
        self.unaligned = unaligned
        self.dataset_name = dataset_name

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*')) # data list
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.files_A[index % len(self.files_A)]
        item_A = sio.loadmat(item_A)
        item_A = item_A['a']
        #item_A = item_A / np.max(item_A)
        item_A = torch.from_numpy(item_A)
        item_A = torch.squeeze(item_A)

        if self.unaligned:          
            item_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            item_B = self.files_B[index % len(self.files_B)]

        item_B = sio.loadmat(item_B)
        item_B = item_B['b']
        #item_B = item_B / np.max(item_B)
        item_B = torch.from_numpy(item_B)
        item_B = torch.squeeze(item_B)
        if self.dataset_name == 'Houston_removal_shadow_3_layers_CNN':
            item_A = item_A.permute(2,0,1)
            item_B = item_B.permute(2,0,1)

        return {'A': item_A, 'B': item_B}

        # A: shadow
        # B: non-shadow

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))