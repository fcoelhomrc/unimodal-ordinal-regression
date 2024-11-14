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
from src.losses import unimodal_wasserstein


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
            padding = base_probas.shape[-1] - 1
            padded_inputs = nn.functional.pad(inputs[:, i], pad=(0, padding), mode='constant', value=0)
            base_probas = convolve(base_probas, padded_inputs, mode="full")
        # base_probas = nn.functional.adaptive_avg_pool1d(base_probas, K)
        base_probas /= base_probas.sum(dim=1, keepdim=True)
        labels = base_probas.argmax(dim=1) // L  # very messy, see test-convolution-approach.py
        return base_probas, labels

class NaiveWassersteinEnsemble(BaseEnsemble):
    """
    Base models should be
    - any (hard or soft) unimodal
    Ensemble should guarantee unimodality
    PROBLEM: Optimized solution is an uniform distribution
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B,
        _, L, K = inputs.shape

        base_probas = inputs[:, 0]  # first base model
        for i in range(1, L):  # wasserstein projection with every other model
            modes = inputs[:, i].argmax(dim=1)
            base_probas /= base_probas.sum(dim=1, keepdim=True)
            base_probas = torch.stack([
                torch.tensor(unimodal_wasserstein(phat, y)[1], dtype=torch.float32, device=inputs.device)
                for phat, y in zip(base_probas.cpu().detach().numpy(), modes.cpu().detach().numpy())
            ])  # need to iterate over batch + use numpy arrays instead of tensors
        labels = base_probas.argmax(dim=1)
        return base_probas, labels

class WassersteinEnsemble(BaseEnsemble):
    """
    Base models should be
    - any (hard or soft) unimodal
    Ensemble should guarantee unimodality
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B,
        _, L, K = inputs.shape

        modes = inputs.argmax(dim=2)
        yproj = torch.stack([
            torch.stack([
                torch.tensor(unimodal_wasserstein(phat / phat.sum(), y)[1], dtype=torch.float32, device=inputs.device)
                for phat, y in zip(inputs[:, i].cpu().detach().numpy(), modes[:, i].cpu().detach().numpy())
            ]) for i in range(L)
        ]).swapdims(0, 1)

        # TODO: remove this...
        debug_projection = False
        if debug_projection:
            nnn = 8
            for i in range(nnn):
                xxx = inputs[i]
                yyy = yproj[i]
                fig, axs = plt.subplots(2, L//2, figsize=(16, 6))
                counter = 0
                for j in range(L):
                    axs.flatten()[counter].plot(xxx[j].cpu().detach().numpy(), label="input")
                    axs.flatten()[counter].plot(yyy[j].cpu().detach().numpy(), label="projection")
                    axs.flatten()[counter].set_title(f'L={j}')
                    axs.flatten()[counter].grid(True)
                    axs.flatten()[counter].legend(loc='best')
                    counter += 1
                fig.suptitle(f'Sample {i}')
                fig.tight_layout()
                fig.show()
            print("done!")


        labels = yproj.argmax(dim=2, keepdim=True)
        return torch.median(labels, dim=1, keepdim=True)[0].squeeze()






if __name__ == "__main__":

    plt.figure()
    losses = [
        "CrossEntropy", "POM", "OrdinalEncoding", "CDW_CE", "UnimodalNet"
    ]

    losses = [
    "CrossEntropy", "POM", "OrdinalEncoding", "CDW_CE", "UnimodalNet",
    "PoissonUnimodal", "VS_SL", "ORD_ACL", "CrossEntropy_UR", "BinomialUnimodal_CE",
    ]

    results = {loss: [] for loss in losses}
    results["Ensemble"] = []
    revs = [1, 2, 3, 4]
    # revs = [1]
    model = WassersteinEnsemble()

    compute_single = False
    for rev in revs:
        # Compute for one at a time
        for loss in losses:
            if not compute_single:
                break
            ds = EnsembleDataset(dataset="FOCUSPATH", loss=[loss], rep=rev)
            batch_size = 32
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

            total_acc = 0
            for x, y in dl:
                ypred = model(x)
                acc = (ypred == y).sum()
                total_acc += acc.item()
            total_acc /= len(ds)
            print(f"{loss} -> {total_acc:3f}")
            results[loss].append(total_acc)

        # Compute with all at same time
        ds = EnsembleDataset(dataset="FOCUSPATH", loss=losses, rep=rev)
        batch_size = 32
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        total_acc = 0
        for x, y in dl:
            ypred = model(x)
            acc = (ypred == y).sum()
            total_acc += acc.item()

        total_acc /= len(ds)
        results["Ensemble"].append(total_acc)

    for key, value in results.items():
        if len(value) > 0:
            value = np.array(value)
            print(f"{key} -> {value.mean():.2f} ± {2 * value.std():.2f}")

