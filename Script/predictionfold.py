import numpy as np
import torch
from torch_geometric.loader import DataLoader
from data import ProteinGraphDataset
from att import AttentionNN
import argparse
import os
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, auc, recall_score, precision_score, f1_score, confusion_matrix, average_precision_score
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='D:/zxm/mine/Example/structure/')
parser.add_argument("--feature_path", type=str, default='D:/zxm/mine/Example/prottrans/')
parser.add_argument('--fasta_file', type=str, default='D:/zxm/mine/Example/test.fasta')
parser.add_argument('--output_prottrans', type=str, default='D:/zxm/mine/Example/prottrans/')
parser.add_argument('--output_esm', type=str, default='D:/zxm/mine/Example/prottrans/')
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

    ece = torch.zeros(1, device=d)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    acc_in_bin = torch.tensor(accuracy_in_bin_list, device=d)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=d)

    return ece, acc_in_bin, avg_conf_in_bin


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

model = AttentionNN(num_node_features, num_classes, dropout=0.3).to(device)
model_path = os.path.normpath(os.path.join(args.output_model, 'best_model_fold_5.pth'))
model.load_state_dict(torch.load(model_path))
model.eval()  

all_preds = []
all_labels = []
all_softmaxes = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        softmax_output = torch.softmax(output, dim=1)
        all_softmaxes.append(softmax_output)

       
        preds = (softmax_output[:, 1] >= 0.5).long().cpu().numpy()  
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu().numpy())


all_labels_tensor = torch.tensor(all_labels).to(device)


all_softmaxes_tensor = torch.cat(all_softmaxes, dim=0)  
ece, acc_in_bin, avg_conf_in_bin = eceloss(all_softmaxes_tensor, all_labels_tensor)


print("Expected Calibration Error (ECE):", ece.item())
print("Accuracy in each bin:", acc_in_bin)
print("Average confidence in each bin:", avg_conf_in_bin)


accuracy = accuracy_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, torch.cat(all_softmaxes, dim=0)[:, 1].cpu().numpy())  
auc_pr = average_precision_score(all_labels, torch.cat(all_softmaxes, dim=0)[:, 1].cpu().numpy()) 
mcc = matthews_corrcoef(all_labels, all_preds)
F1 = f1_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
pre = precision_score(all_labels, all_preds)
balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

print(f'Test Accuracy: {accuracy:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, ROCAUC: {roc_auc:.4f}, AUCPR:{auc_pr:.4f} MCC: {mcc:.4f}, F1: {F1:.4f}, recall: {recall:.4f}, pre: {pre:.4f}')


confusion = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(confusion)

if isinstance(all_labels, list):
    all_labels = torch.tensor(all_labels)
filabels = all_labels.cpu()
if isinstance(all_preds, list):
    all_preds = torch.tensor(all_preds)
fipreds = all_preds.cpu()
fisoftmaxes = all_softmaxes_tensor.cpu()


def calculate_uncertainty(softmaxes):
    """
    Compute uncertainty using the formula: u = p(1-p)/0.25
    """
    uncertainties = (1 - softmaxes[:, 1]) * softmaxes[:, 1] / 0.25
    return uncertainties

def evaluate_uncertainty_threshold(all_labels, all_preds, softmaxes, thresholds=[0.2, 0.4, 0.6, 0.8]):
    """
    Evaluate the model with different uncertainty thresholds.
    Modified from https://github.com/Wang-lab-UCSD/TUnA.
    """
    uncertainties = calculate_uncertainty(softmaxes)
    results = {}

    for cutoff in thresholds:
        filtered_indices = uncertainties < cutoff

        T_filtered = np.array(all_labels)[filtered_indices]
        Y_filtered = np.array(all_preds)[filtered_indices]

        if len(T_filtered) == 0:
            print(f"No samples passed the uncertainty threshold {cutoff}.")
            continue

        accuracy_filtered = accuracy_score(T_filtered, Y_filtered)
        mcc_filtered = matthews_corrcoef(T_filtered, Y_filtered)
        precision_filtered = precision_score(T_filtered, Y_filtered, zero_division=0)
        recall_filtered = recall_score(T_filtered, Y_filtered)
        f1_filtered = f1_score(T_filtered, Y_filtered)
        confusion_filtered = confusion_matrix(T_filtered, Y_filtered)

        results[cutoff] = {
            "Accuracy": accuracy_filtered,
            "MCC": mcc_filtered,
            "Precision": precision_filtered,
            "Recall": recall_filtered,
            "F1 Score": f1_filtered,
            "Confusion Matrix": confusion_filtered
        }

        print(f"Uncertainty Cutoff {cutoff}:")
        print(f"  Accuracy: {accuracy_filtered:.4f}")
        print(f"  MCC: {mcc_filtered:.4f}")
        print(f"  Precision: {precision_filtered:.4f}")
        print(f"  Recall: {recall_filtered:.4f}")
        print(f"  F1 Score: {f1_filtered:.4f}")
        print(f"  Confusion Matrix:\n{confusion_filtered}")

    return results

thresholds = [0.2, 0.4, 0.6, 0.8]
results_scaled = evaluate_uncertainty_threshold(filabels, fipreds, fisoftmaxes, thresholds)
