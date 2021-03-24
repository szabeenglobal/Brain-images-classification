#%%
#make your imports 
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from PIL import Image
import pytorch_lightning as pl
import pandas as pd
import os
import pathlib 
import dataclasses

TRAINDATAPATH = pathlib.Path("/home/dara/Documents/suraiya/data/train_images")
TESTDATAPATH = pathlib.Path("/home/dara/Documents/suraiya/data/test_images")

#%%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.path = path 
        self.df = pd.read_csv("/home/dara/Documents/suraiya/data/train.csv")
        self.df.set_index("ImageId", inplace=True)

        self.file_names = list(path.glob("*.jpg"))

    def __getitem__(self, index):

        path = self.file_names[index] 
        image = Image.open(path)

        image_tensor = torchvision.transforms.functional.to_tensor(image)
        label = self.df.loc[path.name]["ClassId"] 
        
        return image_tensor,label

    def __len__(self):
        return len(self.df)

        

train_dataset = Dataset(TRAINDATAPATH)
test_data = Dataset(TESTDATAPATH)

train_len = int(0.9 * len(train_dataset))
val_len = len(train_dataset) - train_len

train_data, val_data = random_split(train_dataset, [train_len, val_len])

train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=torch.cuda.is_available())
val_data_loader = DataLoader(val_data, batch_size=4, shuffle=False, pin_memory=torch.cuda.is_available())
test_data_loader = DataLoader(test_data, batch_size=4, shuffle=False, pin_memory=torch.cuda.is_available())

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,128, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(128,128),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(128,128),
            torch.nn.Linear(128, 128)
        )

    def forward(self, X):

        return self.layers(X)



    
















#
##%%
#
#class SteelDefects(pl.LightningDataModule): 
#    def __init__(self, TRAINDATAPATH: str, TESTDATAPATH: str, batch_size):
#        super().__init__()
#
#        self.train_data_dir = TRAINDATAPATH
#        self.test_data_dir = TESTDATAPATH
#        self.batch_size = batch_size
#        self.transformation = torchvision.transforms.ToTensor()
#        self.setup()
#
#
#    def setup(self, stage=None):
#        self.steel_defects_test = Dataset(TESTDATAPATH, df)
#        
#        steel_defects_train = Dataset(TRAINDATAPATH, df)
#        
#        self.steel_defects_train, self.steel_defects_val = torch.utils.data.random_split(
#            steel_defects_train, [11000, 1568]
#        )
#
#    def train_dataloader(self):
#        return torch.utils.data.DataLoader(
#            self.steel_defects_train, batch_size=self.batch_size, shuffle=True
#        )
#
#    def val_dataloader(self):
#        return torch.utils.data.DataLoader(self.steel_defects_val, batch_size=self.batch_size)
#
#    def test_dataloader(self):
#        return torch.utils.data.DataLoader(self.steel_defects_test, batch_size=self.batch_size)
#
#
#data_loader = SteelDefects(TRAINDATAPATH, TESTDATAPATH, 500) 
#
## %%
#data_loader.train_dataloader()
## %%
#
