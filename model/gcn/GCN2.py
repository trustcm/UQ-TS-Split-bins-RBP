import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes, dropout, weight_decay):
        super(GCN, self).__init__()
    
        self.conv1 = gnn.GCNConv(num_node_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = gnn.GCNConv(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_transform1 = nn.Linear(num_node_features, 128)

        self.dropout_fc1 = nn.Dropout(dropout) 
        self.fc1 = nn.Linear(64 + 1024+1280, 512)

        self.dropout_fc2 = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.weight_decay = weight_decay

    def forward(self, data):
        x, edge_index, prottrans_feat, esm2 = data.x, data.edge_index, data.prottrans_feat, data.esm2

        esm2 = esm2.to(torch.float32)

        x_residual = self.residual_transform1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)         
        x = F.relu(x)
        x = self.dropout1(x)
        x = x + x_residual

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = torch.cat([x, prottrans_feat,esm2], dim=1)

        x = self.fc1(x)
        x = self.dropout_fc1(x)  
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout_fc2(x)  
        x = F.relu(x)
        x = self.fc3(x)

        return x

