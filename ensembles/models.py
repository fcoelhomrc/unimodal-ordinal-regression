import os
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



class DummyEnsemble(BaseEnsemble):
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        return inputs


# Baselines
class RandomEnsemble(BaseEnsemble):
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs (probas) - batch, base model, class
        # outputs (label) - batch
        B, N, K = inputs.shape
        samples = torch.randint(high=N, size=(B, 1))
        labels = inputs.argmax(dim=2)
        print(labels.shape, samples.shape)
        print(labels, samples)
        return None, labels.gather(1, samples).squeeze()

class MajorityVotingEnsemble(BaseEnsemble):
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs (probas) - batch, base model, class
        # outputs (label) - batch
        B, N, K = inputs.shape
        votes = inputs.argmax(dim=2)
        result = votes.mode(dim=1)
        return None, result.values

class AverageEnsemble(BaseEnsemble):
    """
    Baseline
    Comments: not ordinal
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs (probas) - batch, base model, class
        # outputs (label) - batch
        base_probas = torch.mean(inputs, dim=1)
        labels = base_probas.argmax(dim=1)
        return base_probas, labels




# Ordinal ensembles
class MedianEnsemble(BaseEnsemble):
    """
    Baseline
    Comments: if base models are consistent, ensemble is ordinal
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B
        base_labels = torch.argmax(inputs, dim=2, keepdim=True)
        labels = torch.median(base_labels, dim=1, keepdim=True)[0].squeeze()
        return None, labels

class ConvolutionEnsemble_MovingAverage(BaseEnsemble):
    """
    Moving average strategy
    Base models should be
    - parametric (Binomial, Poisson, etc.) hard unimodal
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B,
        _, L, K = inputs.shape
        base_probas = inputs[:, 0]  # first base model
        for i in range(1, L):  # convolve with every other base model
            # padding = base_probas.shape[-1] - 1
            # padded_inputs = nn.functional.pad(inputs[:, i], pad=(0, padding), mode='constant', value=0)
            base_probas = convolve(base_probas, inputs[:, i], mode="full")  # shape: L * (K - 1) + 1
        kernel_size = (L * K) - (L + K - 2)
        ensemble_probas = nn.functional.avg_pool1d(base_probas.unsqueeze(dim=1),
                                                   kernel_size=kernel_size, stride=1)  # moving average
        ensemble_probas = ensemble_probas.squeeze()
        ensemble_probas /= ensemble_probas.sum(dim=1, keepdim=True)
        labels = ensemble_probas.argmax(dim=1)
        return ensemble_probas, labels

class ConvolutionEnsemble_Stride(BaseEnsemble):
    """
    Stride strategy
    Base models should be
    - parametric (Binomial, Poisson, etc.) hard unimodal
    """
    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B,
        _, L, K = inputs.shape
        base_probas = inputs[:, 0]  # first base model
        for i in range(1, L):  # convolve with every other base model
            base_probas = convolve(base_probas, inputs[:, i], mode="full")
        stride = L
        ensemble_probas =  base_probas[:, ::stride]
        ensemble_probas /= ensemble_probas.sum(dim=1, keepdim=True)
        labels = ensemble_probas.argmax(dim=1)
        return ensemble_probas, labels




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

