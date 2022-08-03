#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from sklearn.metrics import r2_score
from torchvision import models
import matplotlib.pyplot as plt



# In[2]:


batch_size=16
learning_rate=0.003
epochs=20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# In[3]:


# #custum dataset
# # Dataset random split 
# import random

import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor()])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Create train, test, validation set

total_dir ='D:/data2/sars_data_cleaned/minimized/structure_distance/mut-wt/'

label_df = pd.read_csv('C:/Users/JangYuil/Desktop/minimized_distance_dataset220606/processed2.csv')
label_df.set_index('PDB_code', inplace=True)

total_paths=[]
y_total=[]
for mtx in os.listdir(total_dir):
     if mtx[:4] == '6m0j':

         if mtx[:-23].upper() in list(label_df.index):
             total_path=total_dir + mtx
             total_paths.append(total_path)
             y_total.append(label_df.loc[mtx[:-23].upper(), 'ddG'])

 increasing_paths=[]
 y_increasing=[]
 for i in range(len(y_total)):
     if y_total[i] < 0:
         y_increasing.append(y_total[i])
         increasing_paths.append(total_paths[i])

 for path in increasing_paths:
     total_paths.remove(path)
 random.shuffle(total_paths)

 train_paths=total_paths[:3040]
 train_paths.extend(increasing_paths) #trainset에 increasing case몰빵
 test_paths=total_paths[3040:]

 y_train = []
 train_inds = []
 for path in train_paths:
     ind=path.split('/')[-1][:-23].upper()
     label=label_df.loc[ind, 'ddG']
     train_inds.append(ind)
     y_train.append(label)

 y_test = []
 test_inds = []
 for path in test_paths:
     ind=path.split('/')[-1][:-23].upper()
     label=label_df.loc[ind, 'ddG']
     test_inds.append(ind)
     y_test.append(label)
    

    
print("Train size: {}\n Test size: {}".format(len(train_paths), len(test_paths)))



import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor()])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#Define dataset

class CustomDataset(Dataset):
    def __init__(self, data_paths,transform=False):
        self.data_paths = data_paths

        self.transform = transform
        
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_filepath = self.data_paths[idx]
        mtx=pd.read_csv(data_filepath)
        mtx.set_index('Unnamed: 0', inplace=True)
        mtx=np.array(mtx)
        mtx=scaler.fit_transform(mtx)
        mtx = torch.FloatTensor(mtx)

        
        label = float(label_df.loc[data_filepath.split('/')[-1][:-23].upper(), 'ddG'])
        label= torch.tensor(label, dtype=torch.float)
        
                
        return mtx, label
    

#Create Dataset


train_dataset =CustomDataset(train_paths, data_transformer)
test_dataset =CustomDataset(test_paths, data_transformer)



#Define Dataloaders

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader=  DataLoader(
    test_dataset, shuffle=True
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1)
        self.pool =nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)

       
        self.fc1 = nn.Linear(16*5*5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8,1)

    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x=self.pool(x)
        x = F.relu(self.conv3(x))
        x=self.pool(x)

   

    
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
  
  
        
        return x


def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        label=label.view(-1,1)
        optimizer.zero_grad()
        output = model(image.unsqueeze(1))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))



def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    preds=[]
    

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            label=label.view(-1,1)
            output=model(image.unsqueeze(0))
    
            test_loss += criterion(output, label).item()           
            preds.append(output)
            
    
    test_loss /= (len(test_loader.dataset) / batch_size)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    
    return test_loss, test_accuracy, preds



#optimizer, objective function
cnn_model = CNN()
cnn_model.to(device)
optimizer=torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)


# In[560]:



test_accuracies=[]
test_losses=[]
test_preds=[]

torch.manual_seed(42)

for epoch in range(1, epochs + 1):
    train(cnn_model, train_loader, optimizer, log_interval = 20)
    
    test_loss, test_accuracy, test_pred = evaluate(cnn_model, test_loader)
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
    test_preds.append(test_pred)
    scheduler.step() # you can set it like this!
    
    
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))



# save the model
torch.save(cnn_model, 'C:/Users/JangYuil/Desktop/contact_map_cnn/distance_matrix_cnn/cnn_model.pt')






