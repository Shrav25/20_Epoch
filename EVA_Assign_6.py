#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import  numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random


# In[2]:


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# In[3]:


#Define transformers to Normalize the data and convert to tensors
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])


# In[4]:


#Load the dataset
train_data = datasets.MNIST(root='./data',train=True,download=True,transform = transform)
test_data = datasets.MNIST(root='./data',train=False,download=True,transform = transform)


# In[5]:


#Create Dataloaders
train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=False)


# In[6]:


#Build the model
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        
        #First block with BN and dropout
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
#         self.dropout1 = nn.Dropout2d(0.2)
        
        #Second block
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
#         self.dropout2 = nn.Dropout2d(0.3)
        
#         #Third Block
#         self.conv3 = nn.Conv2d(1,8,kernel_size=3,padding=1)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.dropout3 = nn.Dropout2d(0.4)
        
        #Fully connected layer
        self.fc1 = nn.Linear(32*7*7,128)
#         self.bn4 = nn.BatchNorm1d(32)
        
        self.fc2 = nn.Linear(128, 10)  # Output: 10 classes
        self.dropout = nn.Dropout(0.4)
        
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Downsample to 16x16
#         x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # Downsample to 8x8
#         x = self.dropout2(x)
        
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool2d(x, 2)  # Downsample to 4x4
#         x = self.dropout3(x)
        
#         x = x.view(-1, 64 * 4 * 4)  # Flatten for FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# In[7]:


model = MnistModel()
print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# In[8]:


#loss function & Optimise
criterion = nn.CrossEntropyLoss()


# In[9]:


optimizer = optim.Adam(model.parameters(), lr=0.001)  # Correct way


# In[12]:


def eval_model(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to calculate gradients
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
#     return correct, total
    return 100 * correct/total

# correct, total = eval_model(model,test_loader)
# accuracy = 100*correct/total


# In[15]:


for epoch in range(2):
    running_loss = 0.0
    for images, lables in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,lables)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        train_accu = eval_model(model, train_loader)
        test_accu = eval_model(model,test_loader)
        
    print(f'Epoch {epoch+1},loss:{loss.item()}')
    print(f"Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accu:.2f}%, Validation Accuracy: {test_accu:.2f}%")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




