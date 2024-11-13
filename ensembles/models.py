import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torchaudio.functional import convolve

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
    Baseline - always consistent if base models are consistent
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B
        base_labels = torch.argmax(inputs, dim=2, keepdim=True)
        labels = torch.median(base_labels, dim=1, keepdim=True)[0].squeeze()
        return labels

class AverageEnsemble(BaseEnsemble):
    """
    Baseline - averaged proba is not unimodal
    inputs:
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs (probas) - batch, base model, class
        # outputs (label) - batch
        base_probas = torch.mean(inputs, dim=1)
        labels = base_probas.argmax(dim=1)
        return base_probas, labels

class ConvolutionEnsemble(BaseEnsemble):
    """
    Base models should be
    - parametric (Binomial, Poisson, etc.) hard unimodal
    Ensemble should guarantee unimodality (assuming truncating the probas is not a problem...)
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B,
        _, L, K = inputs.shape
        base_probas = inputs[:, 0]  # first base model
        for i in range(1, L):  # convolve with every other base model
            base_probas = convolve(base_probas, inputs[:, i], mode="same")
        base_probas /= base_probas.sum(dim=1, keepdim=True)
        labels = base_probas.argmax(dim=1)
        return base_probas, labels



if __name__ == "__main__":

    plt.figure()

    losses = [
        "BinomialUnimodal_CE", "PoissonUnimodal", "UnimodalNet"
    ]

    results = {loss: [] for loss in losses}
    results["Ensemble"] = []
    # revs = [1, 2, 3, 4]
    revs = [1]
    model = ConvolutionEnsemble()

    compute_single = False
    for rev in revs:
        # Compute for one at a time
        for loss in losses:
            if not compute_single:
                break
            ds = EnsembleDataset(dataset="FOCUSPATH",
                                 loss=[loss],
                                 rev=rev)
            batch_size = 32
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

            total_acc = 0
            for x, y in dl:
                probas, ypred = model(x)
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

        do_single_plot = True
        total_acc = 0
        for x, y in dl:
            probas, ypred = model(x)
            acc = (ypred == y).sum()
            total_acc += acc.item()

            if do_single_plot:
                _, num_models, classes = x.shape
                cls = np.arange(1, classes+1)
                for i in range(num_models):
                    plt.plot(cls, x[0, i].cpu(), label=losses[i], ls="-", lw=1.0, alpha=0.9)
                plt.plot(cls, probas[0].cpu(), label="Ensemble", ls="-", c="k", lw=2.5)

                plt.legend()
                plt.grid()
                plt.xlabel("Classes")
                plt.ylabel("Probability")
                plt.ylim(0, 1.0)
                plt.title("Ensemble v.s. Base Models")
                plt.show()
                do_single_plot = False

        total_acc /= len(ds)
        # print(f"Ensemble -> {total_acc:3f}")
        results["Ensemble"].append(total_acc)

    for key, value in results.items():
        if len(value) > 0:
            value = np.array(value)
            print(f"{key} -> {value.mean():.2f} ± {2 * value.std():.2f}")

