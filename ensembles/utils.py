import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data import FOCUSPATH

ROOT_DIR = os.path.join("..", "predictions")

class EnsembleDataset(Dataset):
    def __init__(self, dataset: str, loss: list, rep: int):
        self.name = dataset.upper()
        self.loss = loss
        self.rev = rep

        self.inputs = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        xx = []
        yy = []
        for i, loss in enumerate(self.loss):
            key = f"{self.name}-{loss}-{self.rev}.pt"
            if not self._is_valid_key(key):
                continue
            x = torch.load(os.path.join(ROOT_DIR, f"pred-{key}"), weights_only=True)
            y = torch.load(os.path.join(ROOT_DIR, f"true-{key}"), weights_only=True)
            xx.append(x); yy.append(y)

        self.inputs = torch.stack(xx)  # shape: L, N, K -> losses, samples, classes
        self.labels = torch.stack(yy)[0, :]  # shape: N -> losses, samples,

    @staticmethod
    def _is_valid_key(key):
        return (os.path.exists(os.path.join(ROOT_DIR, f"pred-{key}")) or
                not os.path.exists(os.path.join(ROOT_DIR, f"true-{key}")))

    def __len__(self):
        return self.inputs.shape[1]

    def __getitem__(self, item):
        return self.inputs[:, item], self.labels[item]



if __name__ == "__main__":

    from src.metrics import *

    ds = EnsembleDataset(dataset="FOCUSPATH", loss=["BinomialUnimodal_CE", "PoissonUnimodal"], rep=1)
    print(len(ds))

    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
