'''
Modified from https://github.com/biomed-AI/nucleic-acid-binding.
'''

import numpy as np
import random
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import os


class BatchSampler(data.Sampler):
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts)) if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not self.batches:
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if not self.batches:
            self._form_batches()
        for batch in self.batches:
            yield batch


class ProteinGraphDataset(data.Dataset):

    def __init__(self, dataset, labels_dict, index, args,
                 top_k=30, device="cuda"):
        super(ProteinGraphDataset, self).__init__()

        self.dataset = {}
        index = set(index)
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())

        self.dataset_path = args.dataset_path
        self.feature_path = args.feature_path
        self.fasta_file = args.fasta_file
        self.output_prottrans = args.output_prottrans
        self.output_esm = args.output_esm
        self.output_dssp = args.output_dssp
        self.output_onehot = args.output_onehot
        self.output_hhm = args.output_hhm
        self.output_pssm = args.output_pssm

        self.top_k = top_k
        self.device = device
        self.node_counts = [len(self.dataset[ID][0]) for ID in self.IDs]

        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        data = self._featurize_as_graph(idx)
        return data

    def _featurize_as_graph(self, idx):
        name = self.IDs[idx]

        coords = np.load(self.dataset_path + name + ".npy")

        coords = torch.as_tensor(coords, device=self.device, dtype=torch.float32)
        
        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        prottrans_feat = torch.tensor(np.load(self.feature_path + name + "_pro.npy"))
        esm_feat = np.load(self.feature_path + name + "_esm.npy", allow_pickle=True)
        if esm_feat.dtype == np.object_:
            esm_feat = np.array([np.array(x, dtype=float) if isinstance(x, np.ndarray) else x for x in esm_feat])
        esm_feat = torch.tensor(esm_feat)
        dssp = torch.tensor(np.load(self.dataset_path + name + "_dssp.npy"))
        onehot = torch.tensor(np.load(self.dataset_path + name + "_onehot.npy"))
        hhm = torch.tensor(np.load(self.dataset_path + name + "_hhm.npy"))
        pssm = torch.tensor(np.load(self.dataset_path + name + "_pssm.npy"))

        node_s = torch.cat([onehot, hhm, pssm, dssp,prottrans_feat,esm_feat], dim=-1).to(torch.float32)


        y_task_str = self.labels_dict[name]
        y = list(map(int, y_task_str))
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)

        data = torch_geometric.data.Data(x=node_s, name=name, y = y )
        return data
