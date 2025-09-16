#%%
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self,data,mode,transforms=None):
        super().__init__()
        self.data = data
        #self.root_dir = 'exercise4/src_to_implement/' #I am not sure where is dataset stored, so I just assume it is here
        self.root_dir = ''
        assert mode in ['val','train','test'] , "Wrong mode selected, must be 'val' or 'train'"
        self.mode = mode
        _transforms = [tv.transforms.ToTensor(),tv.transforms.Normalize(train_mean,train_std)]
        if transforms is not None:
            _transforms.append( transforms)
        self._transform = tv.transforms.Compose(_transforms)

    def __getitem__(self, index):
        img_name = self.data.iloc[index,0]
        label = torch.tensor([int(l) for l in self.data.iloc[index,1:]],dtype=torch.float32)

        #Load image
        img = imread(os.path.join(self.root_dir,img_name))
        img = gray2rgb(img)
        img = self._transform(img)

        return (img,label)
    
    def __len__(self):
        return len(self.data)


    # TODO implement the Dataset class according to the description
    
#%%
if __name__=="__main__":
    
    import pandas as pd

    df = pd.read_csv("data.csv",sep=';')

    

    # dataset = ChallengeDataset(df,mode='train')
    # print(dataset.__len__())
    # a = dataset.__getitem__(0)