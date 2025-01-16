import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(Attention, self).__init__()
        
        self.query_linear = nn.Linear(input_dim, output_dim)
        self.key_linear = nn.Linear(input_dim, output_dim)
        self.value_linear = nn.Linear(input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  
        attention_scores = attention_scores / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32)) 
        attention_weights = torch.softmax(attention_scores, dim=-1) 

        attention_output = torch.matmul(attention_weights, V) 

        attention_output = self.dropout(attention_output)

        output = self.output_linear(attention_output)
        return output

class AttentionNN(nn.Module):
    def __init__(self, num_node_features, num_classes, dropout=0.2):
        super(AttentionNN, self).__init__()

        self.att1 = Attention(input_dim=num_node_features, output_dim=1024, dropout=dropout)
        self.att2 = Attention(input_dim=1024, output_dim=512, dropout=dropout)
        self.att3 = Attention(input_dim=512, output_dim=512, dropout=dropout)

        self.residual1 = nn.Linear(num_node_features, 1024)
        self.residual2 = nn.Linear(1024, 512)
        self.residual3 = nn.Linear(512, 512)

        self.fc1 = nn.Linear(512, 256) 
        self.dropout_fc1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)  
        self.dropout_fc2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, data):
        x = data.x  

        residual = x
        x = self.att1(x)
        x = F.relu(x + self.residual1(residual))  
        
        residual = x
        x = self.att2(x)
        x = F.relu(x + self.residual2(residual))
        
        residual = x
        x = self.att3(x)
        x = F.relu(x + self.residual3(residual))

        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout_fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
