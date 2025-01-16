import numpy as np
import torch
from torch_geometric.loader import DataLoader
from data import ProteinGraphDataset
from att import *
import argparse
import os
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, auc, recall_score, precision_score, f1_score, confusion_matrix, average_precision_score

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='D:/zxm/mine/Example/structure/')
parser.add_argument("--feature_path", type=str, default='D:/zxm/mine/Example/prottrans/')
parser.add_argument('--fasta_file', type=str, default='D:/zxm/mine/Example/test.fasta')
parser.add_argument('--output_prottrans', type=str, default='D:/zxm/mine/Example/prottrans/')
parser.add_argument('--output_esm', type=str, default='D:/zxm/mine/Example/prottrans/')
parser.add_argument('--output_af3', type=str, default='D:/zxm/mine/Example/structure/')
parser.add_argument('--output_dssp', type=str, default='D:/zxm/mine/Example/structure/')
parser.add_argument('--output_onehot', type=str, default='D:/zxm/mine/Example/structure/')
parser.add_argument('--output_pssm', type=str, default='D:/zxm/mine/Example/structure/')
parser.add_argument('--output_hhm', type=str, default='D:/zxm/mine/Example/structure/')

parser.add_argument('--output_model', type=str, default='D:/zxm/mine/Script/attention/Model/end/')

args = parser.parse_args()


def eceloss(softmaxes, labels, n_bins=15):
    """
    Modified from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    d = softmaxes.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []
    bin_samples = []  
    bin_interval_list = []  
    bin_ece_list = []  

    ece = torch.zeros(1, device=d)
    ece_per_bin = []  

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            ece += bin_ece

            bin_interval_list.append((bin_lower.item(), bin_upper.item()))
          
            bin_ece_list.append(bin_ece.item())

            ece_per_bin.append(bin_ece.item())

            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)
            bin_samples.append(in_bin) 

    acc_in_bin = torch.tensor(accuracy_in_bin_list, device=d)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=d)

    return ece, acc_in_bin, avg_conf_in_bin, ece_per_bin, bin_samples, bin_interval_list, bin_ece_list


test_data = {}
test_labels = {}

with open('D:/zxm/mine/Example/RNA-117_Test.fasta') as r1:
    fasta_ori = r1.readlines()

for i in range(0, len(fasta_ori), 3):
    name = fasta_ori[i].strip().split('>')[1]
    seq = fasta_ori[i + 1].strip()
    label = fasta_ori[i + 2].strip()

    test_data[name] = seq
    test_labels[name] = label


test_dataset = ProteinGraphDataset(test_data, test_labels, range(len(test_data)), args)
test_loader = DataLoader(test_dataset, batch_size=16)

num_node_features = 2388
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AttentionNN(num_node_features=num_node_features, num_classes=2, dropout=0.3).to(device)
model_path = os.path.normpath(os.path.join(args.output_model, 'best_model_fold_5.pth'))
model.load_state_dict(torch.load(model_path))
model.eval()  

all_preds = []
all_labels = []
all_softmaxes = []

def apply_temperature_scaling(logits, temperature=0.336):
    """
    Apply temperature scaling to logits.
    """
    return logits / temperature

all_scaled_softmaxes = []


with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        # TS
        scaled_logits = apply_temperature_scaling(output)
        scaled_softmax_output = torch.softmax(scaled_logits, dim=1)
        all_scaled_softmaxes.append(scaled_softmax_output)


        preds = (scaled_softmax_output[:, 1] >= 0.5).long().cpu().numpy()  
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu().numpy())

all_labels_tensor = torch.tensor(all_labels).to(device)

all_scaled_softmaxes_tensor = torch.cat(all_scaled_softmaxes, dim=0)  

#  ECE after TS
ece_scaled, acc_in_bin, avg_conf_in_bin, ece_per_bin, bin_samples, bin_interval_list, bin_ece_list  = eceloss(all_scaled_softmaxes_tensor, all_labels_tensor)
print("Temperature-Scaled ECE:", ece_scaled.item())
for i, (ece_val, bin_interval) in enumerate(zip(ece_per_bin, bin_interval_list)):
                print(f"Bin {i} ECE: {ece_val:.4f}, Interval: ({bin_interval[0]:.4f}, {bin_interval[1]:.4f})")


accuracy = accuracy_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, torch.cat(all_scaled_softmaxes, dim=0)[:, 1].cpu().numpy())  
auc_pr = average_precision_score(all_labels, torch.cat(all_scaled_softmaxes, dim=0)[:, 1].cpu().numpy()) 
mcc = matthews_corrcoef(all_labels, all_preds)
F1 = f1_score(all_labels, all_preds)
pre = precision_score(all_labels, all_preds)
balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

print(f'Test Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}, ROCAUC: {roc_auc:.4f}, AUCPR:{auc_pr:.4f} MCC: {mcc:.4f}, F1: {F1:.4f}',
      f'Precision: {pre:.4f}')

confusion = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(confusion)

# top 2-,4-,6- bin's interval after TS
target_intervals_list = [
    [(0.5333, 0.6), (0.8667, 0.9333)],
    [(0.4667, 0.6667), (0.8667, 0.9333)],
    [(0.4667, 0.8), (0.8667, 0.9333)]
]

for target_intervals in target_intervals_list:
    selected_indices = []
    confidences, _ = torch.max(all_scaled_softmaxes_tensor, 1)

    for interval in target_intervals:
        lower, upper = interval
        in_interval = confidences.gt(lower).logical_and(confidences.le(upper))
        selected_indices.extend(torch.where(in_interval)[0].cpu().numpy().tolist())

    selected_labels = all_labels_tensor[torch.tensor(selected_indices)].cpu().numpy()
    selected_softmaxes = all_scaled_softmaxes_tensor[torch.tensor(selected_indices)]


    print(f"For the selected interval(s): {target_intervals}")

    selected_preds = (selected_softmaxes[:, 1] >= 0.5).long().cpu().numpy()
    pre = precision_score(selected_labels, selected_preds)

    print(f'Precision for selected samples: {pre:.4f}')

    confusion = confusion_matrix(selected_labels, selected_preds)
    print("Confusion Matrix for selected samples:")
    print(confusion)
    print("-" * 50)  