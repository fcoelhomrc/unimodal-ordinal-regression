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
        samples = torch.randint(high=N, size=(B, 1)).to(inputs.device)
        labels = inputs.argmax(dim=2)
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



class WassersteinEnsemble_LP(BaseEnsemble):
    """
    LP solution for Wasserstein barycenter problem
    Base models should be
    - any
    """

    def __init__(self, verbose=False):
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

            # handle ties by randomly selecting a winner
            min_cost = costs.min()
            min_indices = np.where(np.isclose(costs, min_cost))[0]
            # best_mode = np.random.choice(min_indices)
            best_mode = min_indices.mean().astype(int)
            output = torch.tensor(probas[best_mode])
            outputs.append(output)


        ensemble_probas = torch.stack(outputs, dim=0).to(inputs.device)
        labels = ensemble_probas.argmax(dim=1)
        return ensemble_probas, labels


class FastWassersteinEnsemble_LP(BaseEnsemble):
    """
    LP solution for Wasserstein barycenter problem.
    Binary search over modes to speed up inference.
    Base models should be
    - any
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def search_objective(self, mode, probas, cost_matrix, only_cost=True):
        probas, sol = unimodal_wasserstein_barycenter_lp(
                probas.cpu().numpy(),
                cost_matrix,
                mode=mode,
                verbose=self.verbose,
            )
        if only_cost:
            return sol.fun
        else:
            return sol.fun, probas, sol


    def ternary_search(self, base_probas, wasserstein_cost_matrix, K):
        delta = 5  # Total of classes left when we switch to brute force
        left = 0
        right = K - 1
        while right - left > delta:
            m1 = left + (right - left) // 3
            m2 = right - (right - left) // 3

            if (self.search_objective(m1, base_probas, wasserstein_cost_matrix)
                    < self.search_objective(m2, base_probas, wasserstein_cost_matrix)):
                right = m2  # Minimum lies in [left, m2]
            else:
                left = m1  # Minimum lies in [m1, right]

        # Perform brute force on the final small range
        brute_force_range = range(left, right)
        costs = 1e3 * np.ones(K)  # set discarded costs to a very large number, and keep mode<->index correspondence
        probas = np.empty((K, K))
        for mode in brute_force_range:
            cost, proba, sol = self.search_objective(
                mode, base_probas, wasserstein_cost_matrix, only_cost=False
            )
            costs[mode] = cost
            probas[mode] = proba

        # handle ties by randomly selecting a winner
        min_cost = costs.min()
        min_indices = np.where(np.isclose(costs, min_cost))[0]
        # best_mode = np.random.choice(min_indices)
        best_mode = min_indices.mean().astype(int)

        return torch.tensor(probas[best_mode])

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
            output = self.ternary_search(base_probas, wasserstein_cost_matrix, K)
            outputs.append(output)

        ensemble_probas = torch.stack(outputs, dim=0).to(inputs.device)
        labels = ensemble_probas.argmax(dim=1)
        return ensemble_probas, labels
