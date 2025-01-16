import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes, dropout, weight_decay):
        super(MLP, self).__init__()

        combined_feature_dim = num_node_features

        self.dropout_fc1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(combined_feature_dim, 2048)

        self.dropout_fc2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2048, 1024)

        self.dropout_fc3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(1024, 512)

        self.dropout_fc4 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(512, 128)

        self.fc5 = nn.Linear(128, num_classes)

        self.weight_decay = weight_decay

    def forward(self, data):
        x = data.x

        combined_features = x

        x = self.fc1(combined_features)
        x = self.dropout_fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout_fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.dropout_fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.dropout_fc4(x)
        x = F.relu(x)

        x = self.fc5(x)

        return x
