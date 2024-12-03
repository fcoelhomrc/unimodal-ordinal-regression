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

import scipy.spatial.distance
import einops

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

class WassersteinEnsemble_LP(BaseEnsemble):
    """
    LP solution for Wasserstein barycenter problem
    Base models should be
    - any
    """

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        # inputs - B, L, K
        # outputs - B,
        B, L, K = inputs.shape

        outputs = []
        wasserstein_cost_matrix = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(np.arange(K)[:, None])
        )
        for b in range(B):
            base_probas = inputs[b]  # shape: L, K
            base_probas = einops.rearrange(base_probas, "l k -> k l")
            costs = np.zeros(K)
            probas = []
            for mode in range(K):
                proba, sol = unimodal_wasserstein_barycenter_lp(
                    base_probas.cpu().numpy(),
                    wasserstein_cost_matrix,
                    mode=mode,
                    verbose=self.verbose,
                )
                costs[mode] = sol.fun
                probas.append(proba)
            best_mode = costs.argmin()
            outputs.append(torch.tensor(probas[best_mode]))
        ensemble_probas = torch.stack(outputs, dim=0).to(inputs.device)
        labels = ensemble_probas.argmax(dim=1)
        return ensemble_probas, labels



def unimodal_wasserstein_barycenter_lp(A, M, mode, weights=None, verbose=False):
    import scipy.sparse as sps
    import scipy as sp

    def get_A_ub(K, P, mode):
        def get_mode_pattern(K, mode):
            pat = np.zeros((K - 1, K))
            for i in range(mode):
                pat[i, i] = 1
                pat[i, i + 1] = -1
            for i in range(mode, K - 1):
                pat[i, i] = -1
                pat[i, i + 1] = 1
            return pat
        blocks = [
            sps.coo_matrix(
                np.kron(np.ones(K), get_mode_pattern(K, mode))
            )
            for i in range(P)
        ]
        full = sps.block_diag(blocks)
        pad = np.zeros((P * (K - 1), K))
        full = np.concatenate((full.toarray(), pad), axis=1)
        return full

    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert len(weights) == A.shape[1]

    n_distributions = A.shape[1]
    n = A.shape[0]
    n2 = n * n
    c = np.zeros((0))
    b_eq1 = np.zeros((0))
    for i in range(n_distributions):
        c = np.concatenate((c, M.ravel() * weights[i]))
        b_eq1 = np.concatenate((b_eq1, A[:, i]))
    c = np.concatenate((c, np.zeros(n)))
    lst_idiag1 = [sps.kron(sps.eye(n), np.ones((1, n))) for i in range(n_distributions)]
    #  row constraints - eq
    A_eq1 = sps.hstack(
        (sps.block_diag(lst_idiag1), sps.coo_matrix((n_distributions * n, n)))
    )
    # columns constraints - eq
    lst_idiag2 = []
    lst_eye = []
    for i in range(n_distributions):
        if i == 0:
            lst_idiag2.append(sps.kron(np.ones((1, n)), sps.eye(n)))
            lst_eye.append(-sps.eye(n))
        else:
            lst_idiag2.append(sps.kron(np.ones((1, n)), sps.eye(n - 1, n)))
            lst_eye.append(-sps.eye(n - 1, n))
    A_eq2 = sps.hstack((sps.block_diag(lst_idiag2), sps.vstack(lst_eye)))
    b_eq2 = np.zeros((A_eq2.shape[0]))
    # full problem
    A_eq = sps.vstack((A_eq1, A_eq2))
    b_eq = np.concatenate((b_eq1, b_eq2))
    # unimodal part
    # row constraints - ineq
    ## there are no row constraints
    # columns constraints - ineq
    A_ub = get_A_ub(n, n_distributions, mode)
    b_ub = np.zeros(A_ub.shape[0])
    A_ub = sps.coo_matrix(A_ub)

    solver = "highs"
    options = {"disp": verbose}
    sol = sp.optimize.linprog(
        c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        method=solver, options=options
    )
    if sol.success:
        x = sol.x
        b = x[-n:]
        return b, sol
    return None


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

