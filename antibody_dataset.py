import torch
from torch.utils.data import Dataset


class AntibodyDataset(Dataset):
    def __init__(self, sequences, structures, experimental_data):
        self.sequences = sequences
        self.structures = structures
        self.experimental_data = experimental_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.structures[idx], self.experimental_data[idx]
