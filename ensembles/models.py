import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.utils.data import Dataset, DataLoader

import lightning as L

from ensembles.utils import EnsembleDataset


class BaseEnsemble(L.LightningModule):

    def __init__(self):
        super().__init__()

    def setup(self, stage: str) -> None:
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        pass



class MedianEnsemble(BaseEnsemble):
    """
    For models which predict class labels directly
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B, 1, 1
        labels = torch.argmax(inputs, dim=2, keepdim=True)
        return torch.median(labels, dim=1, keepdim=True)[0].squeeze()


class AverageEnsemble(BaseEnsemble):
    """
    Baseline for models which predict class posteriors
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B, 1, 1
        return torch.mean(inputs, dim=1)



if __name__ == "__main__":
    losses = [
        "BinomialUnimodal_CE", "PoissonUnimodal", "CDW_CE",
        "CrossEntropy", "ORD_ACL", "OrdinalEncoding",
        "POM", "UnimodalNet", "VS_SL"
    ]

    results = {loss: [] for loss in losses}
    results["Ensemble"] = []
    revs = [1, 2, 3, 4]
    model = MedianEnsemble()

    for rev in revs:
        # Compute for one at a time
        for loss in losses:
            ds = EnsembleDataset(dataset="FOCUSPATH",
                                 loss=[loss],
                                 rev=rev)
            batch_size = 32
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

            total_acc = 0
            for x, y in dl:
                ypred = model(x)
                acc = (ypred == y).sum()
                total_acc += acc.item()
            total_acc /= len(ds)
            # print(f"{loss} -> {total_acc:3f}")
            results[loss].append(total_acc)

        # Compute with all at same time
        ds = EnsembleDataset(dataset="FOCUSPATH",
                             loss=losses,
                             rev=rev)
        batch_size = 32
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        total_acc = 0
        for x, y in dl:
            ypred = model(x)
            acc = (ypred == y).sum()
            total_acc += acc.item()
        total_acc /= len(ds)
        # print(f"Ensemble -> {total_acc:3f}")
        results["Ensemble"].append(total_acc)

    for key, value in results.items():
        value = np.array(value)
        print(f"{key} -> {value.mean():.2f} ± {2 * value.std():.2f}")

