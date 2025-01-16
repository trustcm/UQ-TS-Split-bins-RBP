import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_node_features, num_classes, dropout, weight_decay):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(2384, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout_fc1 = nn.Dropout(dropout)
        self.dropout_fc2 = nn.Dropout(dropout)

        self.weight_decay = weight_decay

    def forward(self, data):
        x = data.x 
        x = x.unsqueeze(1)  

        x = F.relu(self.conv1(x))      
        x = F.max_pool1d(x, kernel_size=2)  
        x = F.relu(self.conv2(x))  
        x = F.max_pool1d(x, kernel_size=2)  
        x = F.relu(self.conv3(x)) 
        x = F.max_pool1d(x, kernel_size=2)  
        x = x.view(x.size(0), -1)
      
        x = self.fc1(x)  
        x = self.dropout_fc1(x)  
        x = F.relu(x) 

        x = self.fc2(x)  
        x = self.dropout_fc2(x)  
        x = F.relu(x) 

        x = self.fc3(x) 

        return x
