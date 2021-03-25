#%%
#make your imports 
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from PIL import Image
# import pytorch_lightning as pl
import pandas as pd
import os
import pathlib 
import dataclasses

TRAINDATAPATH = pathlib.Path("./data/train_images")
TESTDATAPATH = pathlib.Path("./data/test_images")

#%%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):

        self.path = path 
        self.df = pd.read_csv("./data/train.csv")

        if transform == None: 
            self.transform = transforms.Compose([
                transforms.Resize((400,64)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform


    def __getitem__(self, index):

        path = f"./data/train_images/{self.df.iloc[index, 0]}"
        image = Image.open(path)

        if self.transform: 
            image_tensor = self.transform(image)

        label = self.df.iloc[index]["ClassId"]
        label -= 1
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
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3,
                            stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=None),
            # torch.nn.Conv2d(128, 128, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(128, 128, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(128, 128, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
            # torch.nn.Conv2d(256, 256, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(256, 256, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(256, 256, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.MaxPool2d(kernel_size=2),
            # torch.nn.Conv2d(512, 512, kernel_size=3,
            #                 stride=1, padding=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(512, 512, kernel_size=3,
            #                 stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(409600, 4)
        )                    

    def forward(self, X):
        #this needs to be a range of values, what are we outputting 
        return self.layers(X)  

model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.1)

for batch in train_data_loader:
    
    X, y = batch
    X, y = X.to(device), y.to(device)
    outputs = model(X)
    loss = torch.nn.functional.cross_entropy(outputs, y)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    print(f"this is our loss: {loss.item()}")
    

# %%
