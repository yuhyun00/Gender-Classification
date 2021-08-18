import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import glob
from PIL import Image, ImageFilter

class classification(Dataset):
    
    def __init__(self, select, transform=None):
        super(classification, self).__init__()
        self.select = select
        self.transform = transform
        if select == 0:
            self.data_list = glob.glob('./data/archive/Training/*/*')
            self.label_list = os.listdir('./data/archive/Training/')
        elif select == 1:
            self.data_list = glob.glob('./data/archive/Validation/*/*')
            self.label_list = os.listdir('./data/archive/Validation/')
        elif select == 2:
            self.data_list = glob.glob('./data/archive/Test/*/*')
            self.label_list = os.listdir('./data/archive/Test/')
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        label = self.data_list[idx].split('/')[4]
        label_idx = self.label_list.index(label)
        #img_path = self.data_list[idx].split('train/')[-1]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx #, img_path
