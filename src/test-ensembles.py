# We are following the same recipe as https://arxiv.org/abs/1911.10720

# silence warning when using this without GPU
import warnings

from mpmath import hyper
from tqdm import tqdm
from torch.backends.mkl import verbose

warnings.filterwarnings('ignore', message="Can't initialize NVML")
warnings.filterwarnings('ignore', category=FutureWarning)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('losses')
parser.add_argument('losses_lambda')
parser.add_argument('losses_lambda_values')


parser.add_argument('--datadir', default='predictions')   # TODO: check this
parser.add_argument('--reps', nargs='+', type=int)
parser.add_argument('--only-metric', type=int)

args = parser.parse_args()

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np

import metrics, losses, data
from ensembles.utils import EnsembleDataset
import ensembles.models
# from ensembles.models import AverageEnsemble, MajorityVotingEnsemble, RandomEnsemble
# from ensembles.models import MedianEnsemble, ConvolutionEnsemble_Stride, ConvolutionEnsemble_MovingAverage, WassersteinEnsemble_LP

import os

# save_dir = "../predictions"  # for saving predictions (predicted probas + ground truth label)
# if not os.path.exists(save_dir):
#    os.mkdir(save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## EVAL ##############################

# baselines = [RandomEnsemble(), AverageEnsemble(), MajorityVotingEnsemble()]
# ensembles = [MedianEnsemble(), WassersteinEnsemble_LP(verbose=False)]
# models = baselines + ensembles

try:
    models = [getattr(ensembles.models, args.model)()]
except Exception as e:
    print("Failed to load model: {}".format(e))

def is_proba_available(ensemble):
    no_probas_available = ["RandomEnsemble", "MajorityVotingEnsemble", "MedianEnsemble"]
    name = ensemble.__class__.__name__
    return name not in no_probas_available

def compute_metrics(rep, metrics_list, model):
    losses = args.losses.split(" ")
    losses_lambda = args.losses_lambda.split(" ")
    losses_lambda_values = args.losses_lambda_values.split(" ")

    if losses_lambda == [""]:
        losses_lambda = []
    if losses_lambda_values == [""]:
        losses_lambda_values = []

    hyperparams = [None] * len(losses) + losses_lambda_values
    losses_complete = losses + losses_lambda

    batch_size = 32
    ds = EnsembleDataset(dataset=args.dataset, loss=losses_complete, rep=rep, hyperparam=hyperparams)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    YY_true = []
    YY_pred = []
    PP_pred = []
    for X, Y in dl:
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            P_pred, Yhat = model(X)
        YY_true.append(Y)
        YY_pred.append(Yhat)
        if P_pred is not None:  # some ensembles only make decisions
            PP_pred.append(P_pred)
    YY_true = torch.cat(YY_true)  # ground truth labels
    YY_pred = torch.cat(YY_pred)  # class labels
    if len(PP_pred) > 0:
        PP_pred = torch.cat(PP_pred)  # class probabilities

    metric_values = []
    for metric, needs_probas in metrics_list:
        if needs_probas and not is_proba_available(model):
            metric_values.append(None)
        else:
            m = metric(PP_pred, YY_true, YY_pred)
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            metric_values.append(m)
    return metric_values



# tuples (metric name, needs probabilities)
metrics_list = [
    (metrics.Percentage(metrics.acc), 0),
    (metrics.mae, 0),
    (metrics.Percentage(metrics.quadratic_weighted_kappa), 0),
    (metrics.Percentage(metrics.kendall_tau), 0),
    (metrics.Percentage(metrics.times_unimodal_wasserstein), 1),
    (metrics.Percentage(metrics.wasserstein_metric), 1),
    (metrics.zero_mean_error, 0),
    (metrics.negative_log_likelihood, 1),
]

precisions = [1, 2, 1, 1, 1, 1, 2, 2]

if args.only_metric is not None:
    metrics_list = [metrics_list[args.only_metric]]
    precisions = [precisions[args.only_metric]]


for model in models:
    results_mean = []
    results_std = []

    values_reps = []
    for rep in args.reps:
        values = compute_metrics(rep, metrics_list, model)
        values_reps.append(values)
    values_reps = np.array(values_reps)
    is_none = (values_reps == None)[0]
    mean = values_reps[:, ~is_none].astype(float).mean(axis=0)
    std = values_reps[:, ~is_none].astype(float).std(axis=0)

    counter = 0
    for flag in is_none:
        if flag:
            results_mean.append(None)
            results_std.append(None)
        else:
            results_mean.append(mean[counter])
            results_std.append(std[counter])
            counter += 1
    model_name = model.__class__.__name__
    # print(" ".join(model_name.split("Ensemble")), end="")
    # print(args.dataset, sep=" & ", end="")
    print(' & ' + ' & '.join([f'${mean:.{precision}f}\\color{{gray}}\\pm{std:.{precision}f}$' if mean is not None else
                              f'\\textemdash'
                              for precision, mean, std in zip(precisions, results_mean, results_std)]), end='') # + r' \\')

# results = [compute_metrics(rep, metrics_list) for rep in args.reps]


# print(args.loss.replace('_', r'\_'), end='')
# print(args.dataset, args.loss, sep=' & ', end='')
# if args.print_lambda:
#     print(f' & {args.lamda}', end='')
# if len(args.reps) == 1:
#     print(' & ' + ' & '.join([f'{result:.{precision}f}' for precision, result in zip(precisions, results[0])]), end='') # + r' \\')
#
# else:
#     stds = [torch.std(r[~r.isnan()]) for r in results.T]
#     print(' & ' + ' & '.join([f'${mean:.{precision}f}\\color{{gray}}\\pm{std:.{precision}f}$' for precision, mean, std in zip(precisions, results.nanmean(0), stds)]), end='') # + r' \\')

