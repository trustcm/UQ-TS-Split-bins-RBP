import numpy as np
import os, random
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from data import *
from att import *
import argparse
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
import warnings
import logging
from temperature_scaling import ModelWithTemperature
import torch.optim.lr_scheduler as lr_scheduler


warnings.filterwarnings("ignore", category=DeprecationWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='D:/zxm/mine/train-example/structure/')
parser.add_argument("--feature_path", type=str, default='D:/zxm/mine/train-example/prottrans/')
parser.add_argument('--fasta_file', type=str, default='D:/zxm/mine/train-example/RNA-577-Train.npy')
parser.add_argument('--output_prottrans', type=str, default='D:/zxm/mine/train-example/prottrans/')
parser.add_argument('--output_esm', type=str, default='D:/zxm/mine/train-example/prottrans/')
parser.add_argument('--output_dssp', type=str, default='D:/zxm/mine/train-example/structure/')
parser.add_argument('--output_onehot', type=str, default='D:/zxm/mine/train-example/structure/')
parser.add_argument('--output_hhm', type=str, default='D:/zxm/mine/train-example/structure/')
parser.add_argument('--output_pssm', type=str, default='D:/zxm/mine/train-example/structure/')
parser.add_argument('--output_model', type=str, default='D:/zxm/mine/Script/attention/Model/end/')


args = parser.parse_args()

class FocalLoss_v2(nn.Module):
    def __init__(self, num_class=2, gamma=2, alpha=None):
        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha == None:
            self.alpha = torch.ones(num_class)
        else:
            self.alpha = alpha

    def forward(self, logit, target):
        target = target.view(-1)
        alpha = self.alpha[target.cpu().long()]
        logpt = - F.cross_entropy(logit, target, reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -(alpha * (1 - pt) ** self.gamma) * logpt
        return focal_loss.mean()


train_data = {}
train_labels = {}

with open('D:/zxm/mine/train-example/RNA-495_Train.fasta') as r1:
    fasta_ori = r1.readlines()

for i in range(0, len(fasta_ori), 3):
    name = fasta_ori[i].strip().split('>')[1]
    seq = fasta_ori[i + 1].strip()
    label = fasta_ori[i + 2].strip()
    
    train_data[name] = seq
    label = list(map(int, label))
    train_labels[name] = label


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cross_validation_with_early_stopping(train_data, train_labels, num_folds=5, num_epochs=500, batch_size=16, learning_rate=0.0001, patience=8, seed=42, args=None):

    set_seed(seed)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ProteinGraphDataset(train_data, train_labels, range(len(train_data)), args)

    os.makedirs(args.output_model, exist_ok=True)

    log_path = os.path.join(args.output_model, 'training.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    start_time = time.time()

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{num_folds}')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        print(dataset[0].x.size(1))
       
        model = AttentionNN(num_node_features=dataset[0].x.size(1), num_classes=2, dropout=0.3).to(device)


        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.00001)
        per_cls_weights = torch.FloatTensor([0.2, 0.8]).to(device)
        criterion = FocalLoss_v2(alpha=per_cls_weights, gamma=2)


        best_val_aucroc = float('-inf')
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                
                loss = criterion(output, data.y.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)

            model.eval()
            val_loss = 0.0
            all_val_outputs = []
            all_val_labels = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output = model(data)
                    loss = criterion(output, data.y.long())
                    val_loss += loss.item()
                    all_val_outputs.extend(output.cpu().numpy())
                    all_val_labels.extend(data.y.cpu().numpy())

            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)


            all_val_outputs_np = np.array(all_val_outputs)
            softmax_outputs = torch.softmax(torch.tensor(all_val_outputs_np), dim = 1)
            binary_preds = (softmax_outputs[:, 1] >= 0.5).long().cpu().numpy()
            aucroc = roc_auc_score(all_val_labels, softmax_outputs[:, 1].cpu().numpy())
            Accuracy = sum([1 if pred == label else 0 for pred, label in zip(binary_preds, all_val_labels)]) / len(all_val_labels)
            AUCPRm= average_precision_score(all_val_labels, binary_preds)
            F1 = f1_score(all_val_labels, binary_preds)
            ConfusionMatrix = confusion_matrix(all_val_labels, binary_preds)
            mcc = matthews_corrcoef(all_val_labels, binary_preds)

        
            elapsed_time = time.time() - start_time
            log_message = f"Epoch {epoch + 1} - Elapsed Time: {elapsed_time:.2f}s, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, AUCROC: {aucroc:.4f}, ACC: {Accuracy:.4f}, F1: {F1:.4f}, MCC: {mcc:.4f}, AUCPR: {AUCPRm:.4f} ,Confusion Matrix: {ConfusionMatrix}"
            logging.info(log_message)

            # Early stopping
            if aucroc > best_val_aucroc:
                best_val_aucroc = aucroc
                patience_counter = 0
                best_model_path = os.path.join(args.output_model, f'best_model_fold_{fold + 1}.pth')
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= 8:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

       
results = cross_validation_with_early_stopping(train_data, train_labels, seed=42, args=args)