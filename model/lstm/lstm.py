import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class lstm(nn.Module):
    def __init__(self, num_node_features, num_classes, dropout, weight_decay):
        super(lstm, self).__init__()

        
        self.hidden_size = 256  
        self.num_layers = 2 

        self.bilstm1 = nn.LSTM(num_node_features, self.hidden_size, num_layers=self.num_layers,
                               bidirectional=False, batch_first=True)
        self.bilstm2 = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=self.num_layers,
                               bidirectional=False, batch_first=True)

        self.dropout_fc1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.hidden_size * 2, 256)

        self.dropout_fc2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, num_classes)

        self.weight_decay = weight_decay

    def forward(self, data):
        x = data.x  
        batch_size = x.size(0)  
        seq_length = x.size(1)  

        if len(x.size()) == 2:
            x = x.unsqueeze(1)  

        lstm_out1, (h_n1, _) = self.bilstm1(x)  
        h_n1 = h_n1.permute(1, 0, 2)  
        h_n1 = h_n1.reshape(batch_size, -1)  

        
        h_n1 = h_n1.unsqueeze(1)  
        lstm_out2, (h_n2, _) = self.bilstm2(h_n1)  
        
        h_n2 = h_n2.permute(1, 0, 2).reshape(batch_size, -1)  

        x = self.fc1(h_n2)  
        x = self.dropout_fc1(x)  
        x = F.relu(x)  

        x = self.fc2(x)  
        x = self.dropout_fc2(x)  
        x = F.relu(x)  

        x = self.fc3(x)  

        return x
