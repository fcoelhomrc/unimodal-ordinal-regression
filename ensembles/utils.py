import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATASETS = "data"
ROOT_DIR = os.path.join("..", "predictions")

class EnsembleDataset(Dataset):
    def __init__(self, dataset: str, loss: list, rev: list):
        assert dataset.upper() in ["FOCUSPATH"]
        self.K = getattr(importlib.import_module(DATASETS), "K")
        print(self.K)
        self.name = dataset.upper()
        self.loss = loss
        self.rev = rev

        self.inputs = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        xx = {loss: {rev: [] for rev in self.rev} for loss in self.loss}
        yy = {loss: {rev: [] for rev in self.rev} for loss in self.loss}
        for loss in self.loss:
            for rev in self.rev:
                key = f"{self.name}-{loss}-{rev}.pt"
                if not self._is_valid_key(key):
                    continue
                x = torch.load(os.path.join(ROOT_DIR, f"pred-{key}"), weights_only=True)
                y = torch.load(os.path.join(ROOT_DIR, f"true-{key}"), weights_only=True)
                xx[loss][rev].append(x)
                xx[loss][rev].append(y)

        # shape: N, M, K
        # N - losses (model types)
        # M - revs (folds)
        # K - classes (proba)
        xx = torch.stack([torch.stack([xx[loss][rev] for rev in self.rev]) for loss in self.loss])
        yy = torch.stack([torch.stack([yy[loss][rev] for rev in self.rev]) for loss in self.loss])
        self.inputs = xx
        self.labels = yy

    @staticmethod
    def _is_valid_key(key):
        return (os.path.exists(os.path.join(ROOT_DIR, f"pred-{key}")) or
                not os.path.exists(os.path.join(ROOT_DIR, f"true-{key}")))

    def __len__(self):
        return self.inputs.shape[2]

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]



if __name__ == "__main__":
    ds = EnsembleDataset(dataset="FOCUSPATH",
                         loss=["BinomialUnimodal_CE", "PoissonUnimodal"],
                         rev=[1, 2, 3, 4])
    print(len(ds))

    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for x, y in dl:
        print(x.shape, y.shape)
        break

    print(":)")